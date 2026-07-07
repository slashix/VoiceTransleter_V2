const LANG_NAMES: Record<string, string> = {
  auto: 'auto-detected language',
  en: 'English',
  ru: 'Russian',
  de: 'German',
  fr: 'French',
  es: 'Spanish',
  it: 'Italian',
  pt: 'Portuguese',
  nl: 'Dutch',
  pl: 'Polish',
  ja: 'Japanese',
  ko: 'Korean',
  zh: 'Chinese',
  ar: 'Arabic',
  tr: 'Turkish',
  hi: 'Hindi',
  vi: 'Vietnamese',
  th: 'Thai',
  uk: 'Ukrainian',
  sv: 'Swedish',
  da: 'Danish',
  fi: 'Finnish',
  cs: 'Czech',
  ro: 'Romanian',
  hu: 'Hungarian',
  el: 'Greek',
  he: 'Hebrew',
  id: 'Indonesian',
  ms: 'Malay',
  no: 'Norwegian',
  sk: 'Slovak',
  bg: 'Bulgarian',
  sr: 'Serbian',
  hr: 'Croatian',
  ca: 'Catalan',
  lt: 'Lithuanian',
  lv: 'Latvian',
  et: 'Estonian',
  sl: 'Slovenian'
}

const DEEPSEEK_API_URL = 'https://api.deepseek.com/v1/chat/completions'
const DEEPSEEK_MODEL = process.env.DEEPSEEK_MODEL || 'deepseek-v4-flash'

const SYSTEM_PROMPT = `You are a professional technical translator specializing in dubbing/voiceover scripts for IT/networking content.
Translate the given text accurately, preserving technical terminology and meaning.
Rules:
- Output ONLY the translation, no explanations, no quotes, no notes.
- Keep the tone natural for spoken narration (this will be read aloud by a voice actor).
- For established technical terms and abbreviations (e.g. route reflector, BGP, VXLAN, underlay/overlay, control plane), use the term as commonly used by Russian-speaking network engineers in practice — often transliterated or kept in English (e.g. "route reflector" or "RR", not a literal word-by-word translation), rather than translating literally word-by-word.
- Preserve technical terms, numbers, and proper nouns accurately.
- Do not add or omit information.
- Match the register (formal/informal) of the source text.`

interface DeepSeekChoice {
  message: { content: string }
}

interface DeepSeekResponse {
  choices: DeepSeekChoice[]
  error?: { message: string }
}

function getApiKey(): string {
  const key = process.env.DEEPSEEK_API_KEY
  if (!key) {
    throw new Error(
      'DEEPSEEK_API_KEY не задан. Добавь его в .env или переменные окружения перед запуском.'
    )
  }
  return key
}

async function callDeepSeek(
  text: string,
  sourceLangName: string,
  targetLangName: string,
  attempt = 1
): Promise<string> {
  const maxAttempts = 3
  const apiKey = getApiKey()

  try {
    const response = await fetch(DEEPSEEK_API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${apiKey}`
      },
      body: JSON.stringify({
        model: DEEPSEEK_MODEL,
        messages: [
          { role: 'system', content: SYSTEM_PROMPT },
          {
            role: 'user',
            content: `Translate from ${sourceLangName} to ${targetLangName}:\n\n${text}`
          }
        ],
        temperature: 0.3,
        stream: false
      })
    })

    if (response.status === 503 || response.status === 429) {
      if (attempt < maxAttempts) {
        const backoffMs = 1000 * 2 ** (attempt - 1)
        await new Promise(r => setTimeout(r, backoffMs))
        return callDeepSeek(text, sourceLangName, targetLangName, attempt + 1)
      }
      throw new Error(`DeepSeek API недоступен после ${maxAttempts} попыток (HTTP ${response.status})`)
    }

    if (!response.ok) {
      const errBody = await response.text().catch(() => '')
      throw new Error(`DeepSeek API error ${response.status}: ${errBody}`)
    }

    const data = (await response.json()) as DeepSeekResponse
    if (data.error) {
      throw new Error(`DeepSeek API error: ${data.error.message}`)
    }

    const translated = data.choices?.[0]?.message?.content?.trim()
    if (!translated) {
      throw new Error('DeepSeek вернул пустой ответ')
    }
    return translated
  } catch (e) {
    if (attempt < maxAttempts && e instanceof TypeError) {
      // Сетевая ошибка (обрыв соединения и т.п.) — тоже ретраим
      const backoffMs = 1000 * 2 ** (attempt - 1)
      await new Promise(r => setTimeout(r, backoffMs))
      return callDeepSeek(text, sourceLangName, targetLangName, attempt + 1)
    }
    throw e
  }
}

const BATCH_SYSTEM_PROMPT = `You are a professional technical translator specializing in dubbing/voiceover scripts for IT/networking content.
You will receive a full text broken into numbered parts, each wrapped in markers like [[[PART_1]]] ... text ... [[[/PART_1]]].
Translate the ENTIRE text as one coherent piece, using the full context to keep terminology and tone consistent across parts — this is critical, since the parts are consecutive lines of the same narration.

CRITICAL STRUCTURAL RULES (violating these breaks the downstream pipeline):
- The part boundaries are FIXED and correspond to exact timing slots in a video that will not change. You must NOT move content across part boundaries for any reason — not for better flow, not for grammar, not to avoid a sentence fragment.
- Translate each part's content independently within its own boundaries, even if a sentence started in one part logically continues in the next. Do not merge two parts into one, do not split one part's content into two, do not shift a clause from part N into part N+1 or N-1.
- It is expected and fine for a translated part to end mid-sentence if the source does, or to start with a lowercase continuation — this mirrors the source structure and is required for timing alignment.
- Every input part must have exactly one corresponding output part with the same number, same relative content boundaries, in the same order.
- NEVER translate, alter, remove, renumber, or add markers. Copy them character-for-character exactly as given (e.g. [[[PART_3]]] stays [[[PART_3]]]).
- Output ONLY the translation, with the EXACT SAME marker structure: [[[PART_N]]] ... [[[/PART_N]]] for every part, nothing before the first marker or after the last one, no extra commentary.

EXAMPLE — given this input:
[[[PART_1]]]
Hello, welcome to this tutorial.
[[[/PART_1]]]

[[[PART_2]]]
Today we'll cover BGP route reflectors.
[[[/PART_2]]]

your output (translating English to Russian) must look EXACTLY like this:
[[[PART_1]]]
Привет, добро пожаловать в этот туториал.
[[[/PART_1]]]

[[[PART_2]]]
Сегодня разберём route reflector в BGP.
[[[/PART_2]]]

Note how the markers are byte-for-byte identical to the input, only the text between them is translated.

Rules:
- Keep the tone natural for spoken narration (this will be read aloud by a voice actor).
- For established technical terms and abbreviations (e.g. route reflector, BGP, VXLAN, underlay/overlay, control plane), use the term as commonly used by Russian-speaking network engineers in practice — often transliterated or kept in English (e.g. "route reflector" or "RR", not a literal word-by-word translation), rather than translating literally word-by-word.
- Preserve technical terms, numbers, and proper nouns accurately.
- Do not add or omit information.
- Match the register (formal/informal) of the source text.
- Use consistent terminology for the same concept across all parts — this is the whole point of giving you the full text at once.`

function buildMarkedBlock(texts: string[]): string {
  return texts
    .map((t, i) => `[[[PART_${i + 1}]]]\n${t}\n[[[/PART_${i + 1}]]]`)
    .join('\n\n')
}

function parseMarkedBlock(block: string, expectedCount: number): string[] | null {
  const results: string[] = new Array(expectedCount).fill('')
  const regex = /\[\[\[PART_(\d+)\]\]\]\s*([\s\S]*?)\s*\[\[\[\/PART_\1\]\]\]/g
  let match: RegExpExecArray | null
  let found = 0
  while ((match = regex.exec(block)) !== null) {
    const idx = parseInt(match[1], 10) - 1
    if (idx < 0 || idx >= expectedCount) continue
    results[idx] = match[2].trim()
    found++
  }
  if (found !== expectedCount) return null
  if (results.some(r => r === '')) return null
  return results
}

async function callDeepSeekBatch(
  block: string,
  sourceLangName: string,
  targetLangName: string,
  attempt = 1
): Promise<string> {
  const maxAttempts = 3
  const apiKey = getApiKey()

  try {
    const response = await fetch(DEEPSEEK_API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${apiKey}`
      },
      body: JSON.stringify({
        model: DEEPSEEK_MODEL,
        messages: [
          { role: 'system', content: BATCH_SYSTEM_PROMPT },
          {
            role: 'user',
            content: `Translate from ${sourceLangName} to ${targetLangName}. Preserve markers exactly:\n\n${block}`
          }
        ],
        temperature: 0.3,
        stream: false
      })
    })

    if (response.status === 503 || response.status === 429) {
      if (attempt < maxAttempts) {
        const backoffMs = 1000 * 2 ** (attempt - 1)
        await new Promise(r => setTimeout(r, backoffMs))
        return callDeepSeekBatch(block, sourceLangName, targetLangName, attempt + 1)
      }
      throw new Error(`DeepSeek API недоступен после ${maxAttempts} попыток (HTTP ${response.status})`)
    }

    if (!response.ok) {
      const errBody = await response.text().catch(() => '')
      throw new Error(`DeepSeek API error ${response.status}: ${errBody}`)
    }

    const data = (await response.json()) as DeepSeekResponse
    if (data.error) {
      throw new Error(`DeepSeek API error: ${data.error.message}`)
    }

    const translated = data.choices?.[0]?.message?.content?.trim()
    if (!translated) {
      throw new Error('DeepSeek вернул пустой ответ')
    }
    return translated
  } catch (e) {
    if (attempt < maxAttempts && e instanceof TypeError) {
      const backoffMs = 1000 * 2 ** (attempt - 1)
      await new Promise(r => setTimeout(r, backoffMs))
      return callDeepSeekBatch(block, sourceLangName, targetLangName, attempt + 1)
    }
    throw e
  }
}

/**
 * Переводит массив текстов сегментов ОДНИМ запросом, сохраняя контекст между
 * ними (терминология остаётся согласованной по всему видео), но возвращает
 * результат по частям — ровно столько же элементов, сколько было на входе.
 *
 * Разбивает на батчи, если общий текст слишком длинный (лимит контекста
 * модели и разумный размер одного запроса), и на batch, если DeepSeek
 * всё же нарушит структуру маркеров — тогда для этого батча используется
 * fallback на построчный перевод, чтобы не сорвать весь пайплайн.
 */
export async function translateSegmentsBatch(
  texts: string[],
  sourceLang: string,
  targetLang: string,
  maxBatchChars: number = 6000
): Promise<string[]> {
  if (texts.length === 0) return []

  const sourceLangName = LANG_NAMES[sourceLang] || sourceLang
  const targetLangName = LANG_NAMES[targetLang] || targetLang

  if (sourceLang === targetLang) return texts

  // Группируем сегменты в батчи, чтобы не упереться в лимит контекста
  // на очень длинных видео (сотни сегментов).
  const batches: { texts: string[]; indices: number[] }[] = []
  let current: { texts: string[]; indices: number[] } = { texts: [], indices: [] }
  let currentLen = 0
  texts.forEach((t, i) => {
    const len = t.length + 30 // с запасом на маркеры
    if (currentLen + len > maxBatchChars && current.texts.length > 0) {
      batches.push(current)
      current = { texts: [], indices: [] }
      currentLen = 0
    }
    current.texts.push(t)
    current.indices.push(i)
    currentLen += len
  })
  if (current.texts.length > 0) batches.push(current)

  const result: string[] = new Array(texts.length)

  for (const batch of batches) {
    const block = buildMarkedBlock(batch.texts)
    const responseBlock = await callDeepSeekBatch(block, sourceLangName, targetLangName)
    const parsed = parseMarkedBlock(responseBlock, batch.texts.length)

    if (parsed) {
      batch.indices.forEach((origIdx, i) => { result[origIdx] = parsed[i] })
    } else {
      // Структура маркеров нарушена — откатываемся на построчный перевод
      // для этого батча, чтобы не потерять и не перепутать сегменты.
      for (let i = 0; i < batch.texts.length; i++) {
        const origIdx = batch.indices[i]
        result[origIdx] = await callDeepSeek(batch.texts[i], sourceLangName, targetLangName)
      }
    }
  }

  return result
}

export async function translateText(
  text: string,
  sourceLang: string,
  targetLang: string
): Promise<string> {
  if (!text.trim()) return ''

  const sourceLangName = LANG_NAMES[sourceLang] || sourceLang
  const targetLangName = LANG_NAMES[targetLang] || targetLang

  if (sourceLang === targetLang) return text

  return callDeepSeek(text, sourceLangName, targetLangName)
}
