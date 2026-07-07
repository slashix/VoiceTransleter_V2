import { join } from 'path'
import { existsSync, mkdirSync, writeFileSync, readFileSync } from 'fs'
import { app } from './electron_safe'
import { loadWhisperModel, runWhisper, isWhisperReady } from './whisper'
import { loadTtsModel, synthesizeSpeech, isTtsReady } from './tts'
import { translateText, translateSegmentsBatch } from './translator'
import { getFfmpegPathSafe, runFfmpeg, getVideoDuration } from './ffmpeg'
import { ensurePythonDeps, checkPythonDeps } from './python_deps'
import { buildSyncedAudioTrack, type TimedSegment } from './sync'
import { applyAcronymPronunciation } from './acronyms'

export interface Segment {
  start: number
  end: number
  text: string
  translated?: string
  audioPath?: string
}

const EN_TO_RU: Record<string, string> = {
  sh: 'ш', ch: 'ч', th: 'з', ph: 'ф', gh: 'г', ng: 'нг',
  tion: 'шн', ight: 'айт', ea: 'иа', ou: 'ау', oo: 'у',
  a: 'а', b: 'б', c: 'к', d: 'д', e: 'е', f: 'ф', g: 'г',
  h: 'х', i: 'и', j: 'дж', k: 'к', l: 'л', m: 'м', n: 'н',
  o: 'о', p: 'п', q: 'к', r: 'р', s: 'с', t: 'т', u: 'у',
  v: 'в', w: 'в', x: 'кс', y: 'й', z: 'з'
}

function transliterateEnToRu(text: string): string {
  const lower = text.toLowerCase()
  let result = ''
  let i = 0
  while (i < lower.length) {
    let found = false
    for (let len = 4; len >= 1; len--) {
      const sub = lower.slice(i, i + len)
      if (EN_TO_RU[sub]) {
        result += EN_TO_RU[sub]
        i += len
        found = true
        break
      }
    }
    if (!found) { result += lower[i]; i++ }
  }
  return result
}

function transliterateRuToLa(text: string): string {
  const map: Record<string, string> = {
    а: 'a', б: 'b', в: 'v', г: 'g', д: 'd', е: 'e', ё: 'yo',
    ж: 'zh', з: 'z', и: 'i', й: 'y', к: 'k', л: 'l', м: 'm',
    н: 'n', о: 'o', п: 'p', р: 'r', с: 's', т: 't', у: 'u',
    ф: 'f', х: 'kh', ц: 'ts', ч: 'ch', ш: 'sh', щ: 'shch',
    ъ: '', ы: 'y', ь: '', э: 'e', ю: 'yu', я: 'ya',
    А: 'A', Б: 'B', В: 'V', Г: 'G', Д: 'D', Е: 'E', Ё: 'Yo',
    Ж: 'Zh', З: 'Z', И: 'I', Й: 'Y', К: 'K', Л: 'L', М: 'M',
    Н: 'N', О: 'O', П: 'P', Р: 'R', С: 'S', Т: 'T', У: 'U',
    Ф: 'F', Х: 'Kh', Ц: 'Ts', Ч: 'Ch', Ш: 'Sh', Щ: 'Shch',
    Ъ: '', Ы: 'Y', Ь: '', Э: 'E', Ю: 'Yu', Я: 'Ya'
  }
  return text.split('').map(c => map[c] || c).join('')
}

function formatTimestamp(seconds: number): string {
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  const s = seconds % 60
  return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${s.toFixed(3).padStart(7, '0').replace('.', ',')}`
}

// XTTS v2 молча обрезает/искажает аудио на тексте длиннее лимита, а лимит
// ЗАВИСИТ ОТ ЯЗЫКА — для русского это не общие "250", а 182 символа (см.
// установленный TTS/tts/layers/xtts/tokenizer.py, self.char_limits =
// {"ru": 182, ...}). Дробим по границам предложений, а если отдельное
// предложение всё равно длиннее лимита — по запятым, чтобы не резать
// посреди слов. Без искусственного запаса — applyAcronymPronunciation
// применяется ДО этой функции (см. вызов ниже), значит splitForXtts
// всегда видит уже финальный текст.
const XTTS_CHAR_LIMIT = 182

function splitForXtts(text: string, maxLen: number = XTTS_CHAR_LIMIT): string[] {
  const trimmed = text.trim()
  if (trimmed.length <= maxLen) return [trimmed]

  const sentences = trimmed.split(/(?<=[.!?])\s+/)
  const chunks: string[] = []
  let current = ''
  for (const sent of sentences) {
    const candidate = current ? `${current} ${sent}` : sent
    if (candidate.length <= maxLen) {
      current = candidate
    } else {
      if (current) chunks.push(current)
      current = sent
    }
  }
  if (current) chunks.push(current)

  const final: string[] = []
  for (const chunk of chunks) {
    if (chunk.length <= maxLen) {
      final.push(chunk)
      continue
    }
    const parts = chunk.split(/,\s*/)
    let sub = ''
    for (const p of parts) {
      const candidate = sub ? `${sub}, ${p}` : p
      if (candidate.length <= maxLen) {
        sub = candidate
      } else {
        if (sub) final.push(sub)
        sub = p
      }
    }
    if (sub) final.push(sub)
  }
  return final.length > 0 ? final : [trimmed.slice(0, maxLen)]
}

// Произношение технических акронимов для XTTS (BGP, VXLAN и т.д.) вынесено
// в отдельный файл acronyms.ts — там проще редактировать и дополнять словарь.

// Whisper иногда "залипает" на монотонном/шумном участке аудио и генерирует
// один и тот же текст многократно подряд на крошечных смежных таймкодах
// (известная галлюцинация ASR-моделей). Это даёт десятки секунд дрейфа
// синхронизации на ровном месте — каждая "копия" честно озвучивается
// XTTS заново, но исходный тайм-слот для неё почти нулевой. Убираем такие
// цепочки повторов ДО mergeChoppedSegments, оставляя только первый экземпляр
// с таймингом, растянутым до конца последнего повтора.
function normalizeForDedup(text: string): string {
  return text.trim().toLowerCase().replace(/[.,!?;:'"…]/g, '').replace(/\s+/g, ' ')
}

function dedupeHallucinatedSegments(segments: { start: number; end: number; text: string }[]): { start: number; end: number; text: string }[] {
  if (segments.length === 0) return segments

  const result: { start: number; end: number; text: string }[] = []
  let i = 0
  while (i < segments.length) {
    const current = segments[i]
    const normalized = normalizeForDedup(current.text)
    let j = i + 1
    while (j < segments.length && normalizeForDedup(segments[j].text) === normalized) {
      j++
    }
    const repeatCount = j - i
    if (repeatCount >= 3) {
      // 3+ идущих подряд идентичных сегмента — считаем это галлюцинацией,
      // оставляем один экземпляр с полным таймингом всей цепочки повторов.
      result.push({ start: current.start, end: segments[j - 1].end, text: current.text })
    } else {
      for (let k = i; k < j; k++) result.push(segments[k])
    }
    i = j
  }
  return result
}

// Whisper режет сегменты по внутренним порогам пауз, которые иногда попадают
// ВНУТРЬ одной фразы (например, диктор сделал вдох посреди предложения).
// Это даёт огрызки из 1-3 слов, которые при озвучке XTTS звучат рублено —
// каждый такой кусок получает свою интонацию начала/конца фразы, хотя на
// самом деле это середина одного высказывания. Склеиваем такие соседние
// сегменты обратно в одну фразу перед переводом/синтезом.
const MERGE_GAP_THRESHOLD = 0.35 // сек — пауза короче этого considered "внутри фразы"
const SENTENCE_END_RE = /[.!?…]["')\]]*\s*$/

function mergeChoppedSegments(segments: { start: number; end: number; text: string }[]): { start: number; end: number; text: string }[] {
  if (segments.length === 0) return segments

  const result: { start: number; end: number; text: string }[] = []
  let buffer = { ...segments[0] }

  for (let i = 1; i < segments.length; i++) {
    const seg = segments[i]
    const gap = seg.start - buffer.end
    const bufferEndsSentence = SENTENCE_END_RE.test(buffer.text.trim())

    if (gap <= MERGE_GAP_THRESHOLD && gap >= 0 && !bufferEndsSentence) {
      // Похоже на разрыв внутри одной фразы — склеиваем
      buffer = {
        start: buffer.start,
        end: seg.end,
        text: `${buffer.text.trim()} ${seg.text.trim()}`.replace(/\s+/g, ' ')
      }
    } else {
      result.push(buffer)
      buffer = { ...seg }
    }
  }
  result.push(buffer)

  return result
}

function segmentsToSRt(segments: Segment[], textKey: 'text' | 'translated'): string {
  return segments.map(seg =>
    `${formatTimestamp(seg.start)} --> ${formatTimestamp(seg.end)}\n${seg[textKey] || ''}`
  ).join('\n\n') + '\n'
}function segmentsToPlain(segments: Segment[], textKey: 'text' | 'translated'): string {
  return segments.map(seg => seg[textKey] || '').join('\n')
}

function saveTextFiles(
  segments: Segment[],
  detectedLang: string,
  videoName: string,
  outputDir: string
): void {
  const base = join(outputDir, videoName)

  writeFileSync(base + '_source.txt', segmentsToSRt(segments, 'text'), 'utf8')
  writeFileSync(base + '_source_plain.txt', segmentsToPlain(segments, 'text'), 'utf8')
  writeFileSync(base + '_translation.txt', segmentsToSRt(segments, 'translated'), 'utf8')
  writeFileSync(base + '_translation_plain.txt', segmentsToPlain(segments, 'translated'), 'utf8')

  const translitSource = segments.map(seg => {
    const t = detectedLang === 'ru' ? transliterateRuToLa(seg.text) : transliterateEnToRu(seg.text)
    return { ...seg, text: t }
  })
  writeFileSync(base + '_source_translit.txt', segmentsToSRt(translitSource, 'text'), 'utf8')
  writeFileSync(base + '_source_translit_plain.txt', segmentsToPlain(translitSource, 'text'), 'utf8')

  const translitTranslation = segments.map(seg => {
    const t = transliterateRuToLa(seg.translated || '')
    return { ...seg, translated: t }
  })
  writeFileSync(base + '_translation_translit.txt', segmentsToSRt(translitTranslation, 'translated'), 'utf8')
  writeFileSync(base + '_translation_translit_plain.txt', segmentsToPlain(translitTranslation, 'translated'), 'utf8')
}

const LANG_NAMES: Record<string, string> = {
  en: 'Английский', es: 'Испанский', fr: 'Французский',
  de: 'Немецкий', it: 'Итальянский', pt: 'Португальский',
  nl: 'Нидерландский', pl: 'Польский', ru: 'Русский',
  zh: 'Китайский', ja: 'Японский', ko: 'Корейский',
  ar: 'Арабский', tr: 'Турецкий', hi: 'Хинди',
  vi: 'Вьетнамский', th: 'Тайский', uk: 'Украинский',
  sv: 'Шведский', da: 'Датский', fi: 'Финский',
  cs: 'Чешский', ro: 'Румынский', hu: 'Венгерский',
  el: 'Греческий', he: 'Иврит', id: 'Индонезийский'
}

function getTempDir(clean = false) {
  const dir = join(app.getPath('userData'), 'temp')
  if (clean && existsSync(dir)) {
    const { rmSync } = require('fs')
    try { rmSync(dir, { recursive: true, force: true }) } catch {}
  }
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true })
  return dir
}

function getOutputDir() {
  const dir = join(app.getPath('userData'), 'output')
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true })
  return dir
}

export async function ensureModels(
  onLog: (msg: string) => void,
  onProgress: (pct: number) => void
): Promise<void> {
  onLog('🔍 Проверка зависимостей...')

  // Check and install Python dependencies first
  await ensurePythonDeps(onLog, onProgress)

  if (!getFfmpegPathSafe()) {
    throw new Error('FFmpeg не найден. Нажмите "Загрузить FFmpeg" перед запуском.')
  }

  if (!isWhisperReady()) {
    await loadWhisperModel(onProgress, onLog)
  }

  if (!isTtsReady()) {
    await loadTtsModel(onProgress, onLog)
  }

  onLog('✅ Все модели загружены, запуск пайплайна...')
}

export async function runDubbingPipeline(
  videoPath: string,
  sourceLang: string,
  voiceId: string,
  onLog: (msg: string) => void,
  onProgress: (pct: number) => void
): Promise<string> {
  const tempDir = getTempDir(true)
  const outputDir = getOutputDir()
  const videoName = videoPath.split(/[\\/]/).pop()!.replace(/\.[^.]+$/, '')
  const outputPath = join(outputDir, `${videoName}_dubbed.mp4`)

  onLog('[1/5] Извлечение аудио из видео...')
  onProgress(0.05)
  const audioPath = join(tempDir, 'extracted_audio.wav')
  await runFfmpeg([
    '-y', '-i', videoPath,
    '-vn', '-acodec', 'pcm_s16le',
    '-ar', '16000', '-ac', '1',
    audioPath
  ])
  onLog('  ✅ Аудио извлечено')
  onProgress(0.15)

  onLog('[2/5] Распознавание речи...')
  onProgress(0.2)
  const { segments: rawSegments, detectedLang } = await runWhisper(audioPath, sourceLang, onLog)
  const dedupedSegments = dedupeHallucinatedSegments(rawSegments)
  if (dedupedSegments.length !== rawSegments.length) {
    onLog(`  🔁 Убрано ${rawSegments.length - dedupedSegments.length} повторов (галлюцинация Whisper на монотонном участке)`)
  }
  const segments = mergeChoppedSegments(dedupedSegments)
  if (segments.length !== dedupedSegments.length) {
    onLog(`  🔗 Склеено ${dedupedSegments.length - segments.length} рублёных сегментов (внутрифразовые паузы)`)
  }
  const langLabel = LANG_NAMES[detectedLang] || detectedLang.toUpperCase()
  onLog(`  🌐 Определён язык: ${langLabel} (${detectedLang})`)
  onLog(`  📊 Найдено сегментов: ${segments.length}`)
  onProgress(0.4)

  const isRevoice = detectedLang === 'ru'
  onLog(`[3/5] ${isRevoice ? 'Режим ревойса (язык уже русский)' : 'Перевод на русский (единым блоком с сохранением контекста)...'}`)
  onProgress(0.45)

  let translated: Segment[]
  if (isRevoice) {
    translated = segments.map(seg => ({ ...seg, translated: seg.text }))
  } else {
    const sourceTexts = segments.map(seg => seg.text)
    const translatedTexts = await translateSegmentsBatch(sourceTexts, detectedLang, 'ru')
    translated = segments.map((seg, i) => ({ ...seg, translated: translatedTexts[i] }))
    onLog(`  📝 Переведено ${segments.length}/${segments.length} сегментов (batch-режим, единый контекст)`)
  }
  onProgress(0.6)

  onLog(`[4/5] Синтез речи (XTTS, ~10-15 сек на сегмент)...`)
  onProgress(0.65)
  const ttsSegments: TimedSegment[] = []
  const synthStart = Date.now()
  for (let i = 0; i < translated.length; i++) {
    const seg = translated[i]
    const ttsText = applyAcronymPronunciation(seg.translated!) // только для синтеза — в субтитрах оставляем как есть
    const chunks = splitForXtts(ttsText)
    const txtLen = seg.translated!.length

    if (chunks.length === 1) {
      const segPath = join(tempDir, `seg_${i}.wav`)
      onLog(`  🔊 [${i + 1}/${translated.length}] Синтез ${txtLen} символов... (${seg.translated!.slice(0, 60)}${txtLen > 60 ? '...' : ''})`)
      await synthesizeSpeech(ttsText, segPath, voiceId, onLog)
      ttsSegments.push({ ...seg, audioPath: segPath, actualStart: seg.start })
    } else {
      // Текст сегмента длиннее лимита XTTS (~240 симв.) — дробим на части,
      // синтезируем каждую отдельно и склеиваем в один WAV этого сегмента.
      onLog(`  🔊 [${i + 1}/${translated.length}] Текст ${txtLen} симв. > лимита XTTS, дроблю на ${chunks.length} частей...`)
      const chunkPaths: string[] = []
      for (let c = 0; c < chunks.length; c++) {
        const chunkPath = join(tempDir, `seg_${i}_chunk_${c}.wav`)
        await synthesizeSpeech(chunks[c], chunkPath, voiceId, onLog)
        chunkPaths.push(chunkPath)
      }
      const segPath = join(tempDir, `seg_${i}.wav`)
      await concatWavSegments(chunkPaths, segPath)
      ttsSegments.push({ ...seg, audioPath: segPath, actualStart: seg.start })
    }

    const elapsed = Math.round((Date.now() - synthStart) / 1000)
    onLog(`  ✅ [${i + 1}/${translated.length}] Готово (прошло ${elapsed}с)`)
    onProgress(0.65 + ((i + 1) / translated.length) * 0.15)
  }
  onProgress(0.8)

  onLog('[5/5] Сборка финального видео...')
  onProgress(0.85)
  const videoDur = getVideoDuration(videoPath)
  const syncedAudio = join(tempDir, 'synced.wav')
  await buildSyncedAudioTrack(ttsSegments, syncedAudio, tempDir, videoDur, onLog)
  onLog('  🎵 Аудиодорожка синхронизирована по таймкодам')
  onProgress(0.95)

  await replaceAudioInVideo(videoPath, syncedAudio, outputPath)
  onLog('  🎬 Видео собрано')
  onProgress(1.0)

  onLog('  💾 Сохранение текстовых файлов...')
  saveTextFiles(translated, detectedLang, videoName, outputDir)
  onLog('✅ Готово! Файл сохранён: ' + outputPath)
  return outputPath
}

/**
 * Последовательно склеивает несколько WAV-файлов в один (без пауз) —
 * используется, когда текст сегмента был раздроблен под лимит XTTS
 * и каждая часть синтезирована отдельно.
 */
async function concatWavSegments(inputPaths: string[], outputPath: string): Promise<void> {
  if (inputPaths.length === 1) {
    const { copyFileSync } = require('fs')
    copyFileSync(inputPaths[0], outputPath)
    return
  }
  const listPath = outputPath + '.concat_list.txt'
  const listContent = inputPaths.map(p => `file '${p.replace(/'/g, "'\\''")}'`).join('\n')
  writeFileSync(listPath, listContent, 'utf8')
  await runFfmpeg([
    '-y', '-f', 'concat', '-safe', '0', '-i', listPath,
    '-acodec', 'pcm_s16le', '-ar', '48000', '-ac', '1',
    outputPath
  ])
}

async function replaceAudioInVideo(
  videoPath: string,
  newAudioPath: string,
  outputPath: string
): Promise<void> {
  await runFfmpeg([
    '-y', '-i', videoPath,
    '-i', newAudioPath,
    '-c:v', 'copy',
    '-c:a', 'aac',
    '-b:a', '192k',
    '-map', '0:v:0',
    '-map', '1:a:0',
    '-async', '1',
    outputPath
  ])
}
