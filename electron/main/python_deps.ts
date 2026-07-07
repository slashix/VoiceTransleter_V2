import { execSync, execFileSync, execFile } from 'child_process'
import { promisify } from 'util'
import { existsSync } from 'fs'
import { join } from 'path'
import { app } from 'electron'
import { getPythonCommand } from './platform'

const execFileAsync = promisify(execFile)

/**
 * Check if a Python package is importable.
 * Uses ASYNC execFile (not execFileSync) — this specific check can take
 * 30-90+ seconds on first run (torch/CUDA/cuDNN initialization for heavy
 * packages like TTS), and a synchronous call would block the entire Node
 * event loop, freezing any heartbeat/progress output in a CLI context.
 */
async function isPythonPackageInstalled(pkgName: string): Promise<boolean> {
  const py = getPythonCommand()
  try {
    await execFileAsync(py, ['-c', `import ${pkgName}`], {
      timeout: 90000, // TTS/torch с CUDA-инициализацией может стартовать 30-60+ сек на первом прогоне
      encoding: 'utf-8',
      env: { ...process.env, PYTHONIOENCODING: 'utf-8' },
    })
    return true
  } catch (err: any) {
    console.error(`[isPythonPackageInstalled] '${pkgName}' import failed (py="${py}"): ${err.message}`)
    return false
  }
}

/**
 * Check if a Python package is installed via pip.
 */
function isPipPackageInstalled(pkgName: string): boolean {
  const py = getPythonCommand()
  try {
    const result = execFileSync(py, ['-m', 'pip', 'show', pkgName], {
      stdio: 'pipe',
      timeout: 15000,
      encoding: 'utf-8',
      env: { ...process.env, PYTHONIOENCODING: 'utf-8' },
    })
    return result.includes('Name:')
  } catch {
    return false
  }
}

/**
 * Install a Python package via pip.
 * Logs progress to onLog callback.
 */
async function installPythonPackage(
  pkgName: string,
  onLog?: (msg: string) => void,
  timeout: number = 600000
): Promise<void> {
  const py = getPythonCommand()
  onLog?.(`  📦 Установка ${pkgName}...`)
  try {
    await execFileAsync(py, ['-m', 'pip', 'install', '--upgrade', pkgName], {
      timeout,
      encoding: 'utf-8',
      env: { ...process.env, PYTHONIOENCODING: 'utf-8' },
    })
    onLog?.(`  ✅ ${pkgName} установлен`)
  } catch (err: any) {
    onLog?.(`  ❌ Ошибка установки ${pkgName}: ${err.message.slice(0, 200)}`)
    throw new Error(`Failed to install ${pkgName}`)
  }
}

/**
 * Check if Python itself is available.
 */
export function isPythonAvailable(): boolean {
  const py = getPythonCommand()
  try {
    execFileSync(py, ['--version'], {
      stdio: 'pipe',
      timeout: 5000,
    })
    return true
  } catch {
    return false
  }
}

/**
 * Get the Python version as a string (e.g. "3.11.9").
 */
function getPythonVersion(): string | null {
  const py = getPythonCommand()
  try {
    const output = execFileSync(py, ['--version'], {
      stdio: 'pipe',
      timeout: 5000,
      encoding: 'utf-8',
    })
    // Parse "Python 3.11.9" format
    const match = output.match(/Python\s+(\d+)\.(\d+)\.(\d+)/)
    return match ? `${match[1]}.${match[2]}.${match[3]}` : null
  } catch {
    return null
  }
}

/**
 * Compare semantic versions (e.g. "3.11.9" >= "3.9").
 * Returns: 1 if a > b, 0 if a == b, -1 if a < b
 */
function compareVersions(a: string, b: string): number {
  const pa = a.split('.').map(Number)
  const pb = b.split('.').map(Number)
  const maxLen = Math.max(pa.length, pb.length)
  for (let i = 0; i < maxLen; i++) {
    const va = pa[i] || 0
    const vb = pb[i] || 0
    if (va > vb) return 1
    if (va < vb) return -1
  }
  return 0
}

/**
 * Download and install Python 3.11 using embedded (portable) distribution.
 * No admin rights needed — just unzip and configure pip.
 */
async function downloadAndInstallPython311(
  onLog?: (msg: string) => void,
  onProgress?: (pct: number) => void
): Promise<void> {
  const { existsSync, mkdirSync, rmSync, createWriteStream, writeFileSync, readFileSync } = require('fs')
  const { join } = require('path')
  const { app } = require('electron')
  const https = require('https')
  const { execSync } = require('child_process')

  const pyDir = join(app.getPath('userData'), 'python311')
  const pyExe = join(pyDir, 'python.exe')

  // If already installed, skip
  if (existsSync(pyExe)) {
    onLog?.('  ✅ Python 3.11 уже установлен в папке приложения')
    return
  }

  // Clean up partial install
  if (existsSync(pyDir)) {
    try { rmSync(pyDir, { recursive: true, force: true }) } catch {}
  }
  mkdirSync(pyDir, { recursive: true })

  // Download embedded Python zip (~10 MB)
  const zipUrl = 'https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip'
  const zipPath = join(app.getPath('userData'), 'python-3.11.9-embed.zip')

  onLog?.('  📥 Скачивание Python 3.11.9 (embedded, ~10 МБ)...')
  onProgress?.(0.02)

  await new Promise<void>((resolve, reject) => {
    const download = (url: string, redirects: number = 0) => {
      if (redirects > 5) { reject(new Error('Too many redirects')); return }
      https.get(url, (res: any) => {
        if (res.statusCode === 301 || res.statusCode === 302) {
          download(res.headers.location, redirects + 1)
          return
        }
        if (res.statusCode !== 200) {
          reject(new Error(`HTTP ${res.statusCode}`)); return
        }
        const total = parseInt(res.headers['content-length'] || '0')
        let received = 0
        const file = createWriteStream(zipPath)
        res.on('data', (chunk: Buffer) => {
          received += chunk.length
          if (total > 0) {
            onProgress?.(0.02 + (received / total) * 0.08)
          }
        })
        res.pipe(file)
        file.on('finish', () => { file.close(); resolve() })
        file.on('error', reject)
      }).on('error', reject)
    }
    download(zipUrl)
  })

  onLog?.('  ✅ Скачивание завершено')
  onProgress?.(0.12)

  // Unzip embedded Python
  onLog?.('  📦 Распаковка Python 3.11.9...')
  try {
    // Use PowerShell to unzip (available on all Windows 10+)
    execSync(
      `powershell -Command "Expand-Archive -Path '${zipPath}' -DestinationPath '${pyDir}' -Force"`,
      { stdio: 'pipe', timeout: 60000 }
    )
  } catch (err: any) {
    onLog?.(`  ❌ Ошибка распаковки: ${err.message.slice(0, 200)}`)
    throw new Error('Failed to unzip Python')
  }

  // Verify python.exe
  if (!existsSync(pyExe)) {
    // List what's in the dir for debugging
    try {
      const { readdirSync } = require('fs')
      const files = readdirSync(pyDir)
      onLog?.(`  ⚠️ python.exe не найден. Содержимое папки: ${files.slice(0, 10).join(', ')}`)
    } catch {}
    throw new Error('Python.exe not found after unzip')
  }

  // Clean up zip
  try { rmSync(zipPath, { force: true }) } catch {}

  // Enable site-packages (needed for pip)
  const pthFile = join(pyDir, 'python311._pth')
  if (existsSync(pthFile)) {
    let content = readFileSync(pthFile, 'utf-8')
    // Uncomment "import site" line
    content = content.replace('#import site', 'import site')
    writeFileSync(pthFile, content)
  }

  // Download and install pip via get-pip.py
  onLog?.('  📦 Установка pip...')
  const getPipPath = join(pyDir, 'get-pip.py')
  await new Promise<void>((resolve, reject) => {
    https.get('https://bootstrap.pypa.io/get-pip.py', (res: any) => {
      if (res.statusCode !== 200) { reject(new Error(`HTTP ${res.statusCode}`)); return }
      const file = createWriteStream(getPipPath)
      res.pipe(file)
      file.on('finish', () => { file.close(); resolve() })
      file.on('error', reject)
    }).on('error', reject)
  })

  try {
    execSync(`"${pyExe}" "${getPipPath}" --no-warn-script-location`, {
      stdio: 'pipe',
      timeout: 120000,
      env: { ...process.env, PYTHONIOENCODING: 'utf-8' },
    })
  } catch (err: any) {
    onLog?.(`  ❌ Ошибка установки pip: ${err.message.slice(0, 200)}`)
    throw new Error('Failed to install pip')
  }

  try { rmSync(getPipPath, { force: true }) } catch {}

  // Install faster-whisper
  onLog?.('  📦 Установка faster-whisper...')
  onProgress?.(0.15)
  try {
    execSync(`"${pyExe}" -m pip install --no-warn-script-location faster-whisper`, {
      stdio: 'pipe',
      timeout: 600000,
      env: { ...process.env, PYTHONIOENCODING: 'utf-8' },
    })
    onLog?.('  ✅ faster-whisper установлен')
  } catch {
    throw new Error('Failed to install faster-whisper')
  }

  // Pre-install TTS build dependencies (numpy, cython, torch needed for TTS build)
  onLog?.('  📦 Установка зависимостей для TTS...')
  try {
    execSync(`"${pyExe}" -m pip install --no-warn-script-location numpy cython`, {
      stdio: 'pipe',
      timeout: 300000,
      env: { ...process.env, PYTHONIOENCODING: 'utf-8' },
    })
  } catch {}

  // Install TTS engine — coqui-tts (МОДЕРНИЗИРОВАННЫЙ ФОРК original TTS,
  // который не поддерживается с 2024 года и не устанавливается на новых
  // версиях Python/зависимостей). Оригинальный код здесь ставил "TTS" —
  // заменено на "coqui-tts", чтобы Windows embedded-Python путь был
  // консистентен с основным путём установки (см. isPythonPackageInstalled
  // ниже, которая проверяет тот же самый пакет через import TTS — оба
  // называют python-модуль одинаково, различается только имя pip-пакета).
  onLog?.('  📦 Установка TTS-движка (coqui-tts, ~5-10 мин)...')
  try {
    const ttsResult = execSync(
      `"${pyExe}" -m pip install --no-warn-script-location "coqui-tts[codec]"`,
      {
        stdio: 'pipe',
        timeout: 1800000, // 30 minutes
        env: { ...process.env, PYTHONIOENCODING: 'utf-8', PIP_DEFAULT_TIMEOUT: '300' },
        encoding: 'utf-8',
      }
    )
    onLog?.('  ✅ coqui-tts установлен')

    // transformers>=5.1 несовместим с coqui-tts (issue idiap/coqui-ai-TTS#558)
    onLog?.('  📦 Фиксация версии transformers (совместимость с coqui-tts)...')
    execSync(`"${pyExe}" -m pip install --no-warn-script-location "transformers==4.57.6"`, {
      stdio: 'pipe',
      timeout: 300000,
      env: { ...process.env, PYTHONIOENCODING: 'utf-8' },
    })
    onLog?.('  ✅ transformers зафиксирован на 4.57.6')
  } catch (err: any) {
    // Capture stderr for debugging
    const stderr = err.stderr?.toString() || ''
    const stdout = err.stdout?.toString() || ''
    onLog?.(`  ⚠️ stderr: ${stderr.slice(-500)}`)
    onLog?.(`  ⚠️ stdout: ${stdout.slice(-500)}`)

    // Try alternative: install with --no-deps then deps separately
    onLog?.('  🔄 Повторная установка coqui-tts (--no-build-isolation)...')
    try {
      execSync(`"${pyExe}" -m pip install --no-warn-script-location --no-build-isolation "coqui-tts[codec]"`, {
        stdio: 'pipe',
        timeout: 1800000,
        env: { ...process.env, PYTHONIOENCODING: 'utf-8', PIP_DEFAULT_TIMEOUT: '300' },
      })
      onLog?.('  ✅ coqui-tts установлен (повтор)')
    } catch (err2: any) {
      const stderr2 = err2.stderr?.toString() || ''
      onLog?.(`  ❌ Ошибка coqui-tts: ${stderr2.slice(-500)}`)
      throw new Error(`Failed to install coqui-tts: ${stderr2.slice(-300)}`)
    }
  }

  onProgress?.(0.2)
  onLog?.('  ✅ Python 3.11.9 + все зависимости установлены!')
}

/**
 * Check if Python version is compatible.
 * Coqui TTS (fork "coqui-tts" on PyPI) requires Python >=3.9,<3.13.
 * (Original unmaintained "TTS" package on PyPI is capped at <3.12 — we use
 * the maintained fork instead, which supports 3.12 as well.)
 */
function isPythonVersionCompatible(): { ok: boolean; version: string; reason?: string } {
  const version = getPythonVersion()
  if (!version) {
    return { ok: false, version: 'unknown', reason: 'Не удалось определить версию Python' }
  }
  if (compareVersions(version, '3.9') < 0) {
    return { ok: false, version, reason: `Python ${version} слишком старый. Требуется Python 3.9–3.12. Скачайте с python.org` }
  }
  if (compareVersions(version, '3.13') >= 0) {
    return { ok: false, version, reason: `Python ${version} не поддерживается. Coqui TTS требует Python 3.9–3.12. Установите Python 3.12 с python.org` }
  }
  return { ok: true, version }
}

/**
 * Check if all required Python packages are installed.
 */
export interface PythonDepsStatus {
  python: boolean
  pythonVersion: string
  pythonCompatible: boolean
  fasterWhisper: boolean
  tts: boolean
}

export async function checkPythonDeps(): Promise<PythonDepsStatus> {
  const pythonOk = isPythonAvailable()
  if (!pythonOk) {
    return { python: false, pythonVersion: '', pythonCompatible: false, fasterWhisper: false, tts: false }
  }
  const versionCheck = isPythonVersionCompatible()
  const fasterWhisper = versionCheck.ok && await isPythonPackageInstalled('faster_whisper')
  const tts = versionCheck.ok && await isPythonPackageInstalled('TTS')
  return {
    python: true,
    pythonVersion: versionCheck.version,
    pythonCompatible: versionCheck.ok,
    fasterWhisper,
    tts,
  }
}

/**
 * Ensure all required Python packages are installed.
 * Installs missing ones automatically.
 */
export async function ensurePythonDeps(
  onLog?: (msg: string) => void,
  onProgress?: (pct: number) => void
): Promise<void> {
  const status = await checkPythonDeps()

  if (!status.python) {
    // Python not found at all — try auto-install on Windows
    if (process.platform === 'win32') {
      onLog?.('📦 Python не найден. Автоустановка Python 3.11...')
      try {
        await downloadAndInstallPython311(onLog, onProgress)
        const { join } = require('path')
        const { app } = require('electron')
        const pyExe = join(app.getPath('userData'), 'python311', 'python.exe')
        process.env.VOICE_TRANSLATOR_PYTHON = pyExe
        onLog?.('✅ Python 3.11 установлен и готов к работе')
        return
      } catch (err: any) {
        throw new Error(
          `Не удалось установить Python 3.11 автоматически: ${err.message}\n` +
          `Установите Python 3.11 вручную с python.org`
        )
      }
    }
    throw new Error(
      'Python не найден в системе. Установите Python 3.10–3.11 с python.org и попробуйте снова.'
    )
  }

  if (!status.pythonCompatible) {
    // Try to auto-install Python 3.11 (Windows only)
    if (process.platform === 'win32') {
      onLog?.(`⚠️ Python ${status.pythonVersion} несовместим. Автоустановка Python 3.11...`)
      try {
        await downloadAndInstallPython311(onLog, onProgress)
        // Update environment to use embedded Python
        const { join } = require('path')
        const { app } = require('electron')
        const pyExe = join(app.getPath('userData'), 'python311', 'python.exe')
        process.env.VOICE_TRANSLATOR_PYTHON = pyExe
        onLog?.('✅ Python 3.11 установлен и готов к работе')
        // Re-check with new Python
        const newStatus = await checkPythonDeps()
        if (!newStatus.python) {
          throw new Error('Установленный Python 3.11 не найден')
        }
        return
      } catch (err: any) {
        throw new Error(
          `Не удалось установить Python 3.11 автоматически: ${err.message}\n` +
          `Установите Python 3.11 вручную с python.org`
        )
      }
    } else {
      const versionCheck = isPythonVersionCompatible()
      throw new Error(
        versionCheck.reason ||
          `Python ${status.pythonVersion} не поддерживается. Установите Python 3.10 или 3.11 с python.org.`
      )
    }
  }

  const missing: string[] = []
  if (!status.fasterWhisper) missing.push('faster-whisper')
  if (!status.tts) missing.push('TTS')

  if (missing.length === 0) {
    onLog?.(`✅ Python-зависимости установлены (Python ${status.pythonVersion})`)
    return
  }

  onLog?.(`📦 Установка Python-зависимостей: ${missing.join(', ')}...`)
  onProgress?.(0.05)

  // Оригинальный пакет "TTS" на PyPI не поддерживается с 2024 года и не
  // ставится на актуальных версиях зависимостей — используем поддерживаемый
  // форк "coqui-tts". Имя для проверки (import TTS — модуль внутри пакета
  // называется одинаково у обоих) отличается от имени для pip install.
  const PIP_PACKAGE_NAME: Record<string, string> = {
    'TTS': 'coqui-tts[codec]',
  }

  const total = missing.length
  let installed = 0

  for (const pkg of missing) {
    const pipName = PIP_PACKAGE_NAME[pkg] ?? pkg
    await installPythonPackage(pipName, onLog)
    if (pkg === 'TTS') {
      // transformers>=5.1 несовместим с coqui-tts (issue idiap/coqui-ai-TTS#558)
      onLog?.('  📦 Фиксация версии transformers (совместимость с coqui-tts)...')
      await installPythonPackage('transformers==4.57.6', onLog)
    }
    installed++
    onProgress?.(0.05 + (installed / total) * 0.15)
  }

  onLog?.(`✅ Все Python-зависимости установлены (Python ${status.pythonVersion})`)
  onProgress?.(0.2)
}
