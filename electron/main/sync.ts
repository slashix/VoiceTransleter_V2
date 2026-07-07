import { join } from 'path'
import { readFileSync, writeFileSync } from 'fs'
import { getFfmpegPathSafe, runFfmpeg } from './ffmpeg'
import type { Segment } from './pipeline'

// ---- Настройки синхронизации -------------------------------------------
// Если TTS-сегмент короче своего тайм-слота на видео:
//   - разницу до SOFT_STRETCH_LIMIT (15%) компенсируем растяжением (atempo < 1.0),
//     речь звучит чуть медленнее, но естественно
//   - остаток свыше этого — тишина в конце сегмента
const SOFT_STRETCH_LIMIT = 0.15

// Если TTS-сегмент длиннее своего тайм-слота:
//   - сжимаем через atempo, но не быстрее чем во столько раз
const MAX_COMPRESSION_RATIO = 1.5

const OUTPUT_SAMPLE_RATE = 48000

export interface TimedSegment extends Segment {
  audioPath: string
  /** Реальная позиция начала на итоговой дорожке (после накопленных сдвигов) */
  actualStart: number
}

/**
 * Читает длительность WAV-файла напрямую из заголовка (без ffprobe, т.к. это
 * наши собственные короткие TTS-сегменты, а не произвольные внешние файлы).
 */
function readWavDuration(wavPath: string): number {
  const buf = readFileSync(wavPath)
  const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength)
  const sampleRate = view.getUint32(24, true)
  let dataSize = 0
  let offset = 12
  while (offset + 8 <= buf.length) {
    const id = String.fromCharCode(
      view.getUint8(offset), view.getUint8(offset + 1),
      view.getUint8(offset + 2), view.getUint8(offset + 3)
    )
    const sz = view.getUint32(offset + 4, true)
    if (id === 'data') { dataSize = sz; break }
    offset += 8 + sz + (sz % 2)
  }
  return dataSize / 2 / sampleRate
}

/**
 * Строит ffmpeg atempo-цепочку для произвольного коэффициента,
 * т.к. один фильтр atempo поддерживает только диапазон [0.5, 2.0].
 */
function buildAtempoChain(ratio: number): string[] {
  const filters: string[] = []
  let r = ratio
  while (r > 2.0) { filters.push('atempo=2.0'); r /= 2.0 }
  while (r < 0.5) { filters.push('atempo=0.5'); r /= 0.5 }
  filters.push(`atempo=${r.toFixed(4)}`)
  return filters
}

/**
 * Подгоняет один TTS-сегмент под целевую длительность (тайм-слот на видео).
 * Возвращает путь к обработанному WAV и фактическую длительность результата.
 */
async function fitSegmentToSlot(
  inputPath: string,
  outputPath: string,
  currentDur: number,
  targetDur: number,
  onLog?: (msg: string) => void
): Promise<{ path: string; duration: number }> {
  if (targetDur <= 0.01) {
    // Вырожденный случай — почти нулевой слот, просто копируем как есть
    const { copyFileSync } = require('fs')
    copyFileSync(inputPath, outputPath)
    return { path: outputPath, duration: currentDur }
  }

  const diff = targetDur - currentDur // >0 — сегмент короче слота, <0 — длиннее

  if (Math.abs(diff) < 0.02) {
    const { copyFileSync } = require('fs')
    copyFileSync(inputPath, outputPath)
    return { path: outputPath, duration: currentDur }
  }

  if (diff > 0) {
    // Сегмент короче слота
    const relDiff = diff / targetDur
    if (relDiff <= SOFT_STRETCH_LIMIT) {
      // Полностью гасим растяжением
      const ratio = currentDur / targetDur // <1.0 => atempo замедлит
      const filters = buildAtempoChain(ratio)
      await runFfmpeg([
        '-y', '-i', inputPath,
        '-af', filters.join(','),
        '-acodec', 'pcm_s16le', '-ar', String(OUTPUT_SAMPLE_RATE), '-ac', '1',
        outputPath
      ])
      return { path: outputPath, duration: targetDur }
    } else {
      // Тянем только до предела SOFT_STRETCH_LIMIT, остаток — тишина в конце
      const stretchedTargetDur = currentDur / (1 - SOFT_STRETCH_LIMIT)
      const ratio = currentDur / stretchedTargetDur
      const filters = buildAtempoChain(ratio)
      const silenceTail = targetDur - stretchedTargetDur

      const stretchedTmp = outputPath + '.stretch.wav'
      await runFfmpeg([
        '-y', '-i', inputPath,
        '-af', filters.join(','),
        '-acodec', 'pcm_s16le', '-ar', String(OUTPUT_SAMPLE_RATE), '-ac', '1',
        stretchedTmp
      ])
      // Добавляем тишину в конец через apad (по времени)
      await runFfmpeg([
        '-y', '-i', stretchedTmp,
        '-af', `apad=pad_dur=${silenceTail.toFixed(3)}`,
        '-acodec', 'pcm_s16le', '-ar', String(OUTPUT_SAMPLE_RATE), '-ac', '1',
        outputPath
      ])
      onLog?.(`    ↳ сегмент короче на ${(diff).toFixed(2)}с, растянут до предела ${(SOFT_STRETCH_LIMIT * 100).toFixed(0)}%, добавлена тишина ${silenceTail.toFixed(2)}с`)
      return { path: outputPath, duration: targetDur }
    }
  } else {
    // Сегмент длиннее слота — сжимаем, но не более MAX_COMPRESSION_RATIO
    const neededRatio = currentDur / targetDur // >1.0
    const appliedRatio = Math.min(neededRatio, MAX_COMPRESSION_RATIO)
    const filters = buildAtempoChain(appliedRatio)
    await runFfmpeg([
      '-y', '-i', inputPath,
      '-af', filters.join(','),
      '-acodec', 'pcm_s16le', '-ar', String(OUTPUT_SAMPLE_RATE), '-ac', '1',
      outputPath
    ])
    const resultDur = currentDur / appliedRatio
    if (neededRatio > MAX_COMPRESSION_RATIO) {
      const overflow = resultDur - targetDur
      onLog?.(`    ⚠️ сегмент длиннее слота, сжат до предела ${MAX_COMPRESSION_RATIO}x, всё равно превышение ${overflow.toFixed(2)}с — сдвигаем следующий сегмент`)
    }
    return { path: outputPath, duration: resultDur }
  }
}

/**
 * Собирает финальную аудиодорожку из TTS-сегментов, размещая каждый строго
 * по своему таймкоду на видео (по данным Whisper), с индивидуальной
 * подгонкой длительности вместо глобального time-stretch всей дорожки.
 */
export async function buildSyncedAudioTrack(
  segments: TimedSegment[],
  outputPath: string,
  tempDir: string,
  totalVideoDuration: number,
  onLog?: (msg: string) => void
): Promise<void> {
  if (!getFfmpegPathSafe()) throw new Error('FFmpeg not found')
  if (segments.length === 0) {
    throw new Error('Нет сегментов для сборки аудиодорожки')
  }

  const fittedPaths: string[] = []
  const fittedStarts: number[] = []
  let timeDrift = 0 // накопленный сдвиг из-за сегментов, упёршихся в MAX_COMPRESSION_RATIO

  for (let i = 0; i < segments.length; i++) {
    const seg = segments[i]
    const slotStart = seg.start + timeDrift
    const nextOriginalStart = i + 1 < segments.length ? segments[i + 1].start : totalVideoDuration
    const slotEnd = Math.max(slotStart, nextOriginalStart + timeDrift)
    const targetDur = Math.max(0.05, slotEnd - slotStart)

    const currentDur = readWavDuration(seg.audioPath)
    const fittedPath = join(tempDir, `fitted_${i}.wav`)

    onLog?.(`  ⏱ [${i + 1}/${segments.length}] слот ${targetDur.toFixed(2)}с, синтез ${currentDur.toFixed(2)}с`)

    const { duration: actualDur } = await fitSegmentToSlot(
      seg.audioPath, fittedPath, currentDur, targetDur, onLog
    )

    // Если после сжатия по максимуму сегмент всё равно не влез — копим drift,
    // сдвигая последующие сегменты вперёд, чтобы не терять речь обрезкой
    const overflow = actualDur - targetDur
    if (overflow > 0.02) {
      timeDrift += overflow
    }

    fittedPaths.push(fittedPath)
    fittedStarts.push(slotStart)
  }

  if (timeDrift > 1.0) {
    onLog?.(`  ⚠️ Накопленный сдвиг тайминга к концу видео: ${timeDrift.toFixed(2)}с (из-за сегментов, упиравшихся в лимит сжатия ${MAX_COMPRESSION_RATIO}x)`)
  }

  // Собираем итоговую дорожку через ffmpeg: тишина нужной длины + каждый
  // подогнанный сегмент, размещённый через adelay на своей позиции, все
  // входы микшируются amix.
  const totalDur = Math.max(
    totalVideoDuration + timeDrift,
    fittedStarts[fittedStarts.length - 1] + readWavDuration(fittedPaths[fittedPaths.length - 1])
  )

  const inputArgs: string[] = []
  const filterParts: string[] = []
  fittedPaths.forEach((p, i) => {
    inputArgs.push('-i', p)
    const delayMs = Math.round(fittedStarts[i] * 1000)
    filterParts.push(`[${i}:a]adelay=${delayMs}|${delayMs}[a${i}]`)
  })
  const mixInputs = fittedPaths.map((_, i) => `[a${i}]`).join('')
  const filterComplex =
    filterParts.join(';') +
    `;${mixInputs}amix=inputs=${fittedPaths.length}:duration=longest:normalize=0[out]`

  await runFfmpeg([
    '-y',
    ...inputArgs,
    '-filter_complex', filterComplex,
    '-map', '[out]',
    '-t', totalDur.toFixed(3),
    '-acodec', 'pcm_s16le', '-ar', String(OUTPUT_SAMPLE_RATE), '-ac', '1',
    outputPath
  ])
}
