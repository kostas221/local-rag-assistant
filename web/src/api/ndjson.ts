/**
 * Ανάγνωση NDJSON (ένα JSON ανά γραμμή) από ροή, όπως τη στέλνει το /chat.
 *
 * Τα πακέτα του δικτύου δεν σέβονται τις γραμμές: μία γραμμή μπορεί να έρθει σε δύο κομμάτια, και ένας ελληνικός
 * χαρακτήρας (2 bytes) μπορεί να κοπεί στη μέση. Γι' αυτό: TextDecoder με `stream: true` και buffer μέχρι το `\n`.
 */
export class NdjsonError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'NdjsonError'
  }
}

function parseLine(line: string): unknown {
  try {
    return JSON.parse(line)
  } catch {
    throw new NdjsonError(`Invalid line in the response stream: ${line.slice(0, 80)}`)
  }
}

export async function* readNdjson(body: ReadableStream<Uint8Array>): AsyncGenerator<unknown> {
  const reader = body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  try {
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      let newline = buffer.indexOf('\n')
      while (newline >= 0) {
        const line = buffer.slice(0, newline).trim()
        buffer = buffer.slice(newline + 1)
        if (line) yield parseLine(line)
        newline = buffer.indexOf('\n')
      }
    }
    buffer += decoder.decode()
    const last = buffer.trim()
    if (last) yield parseLine(last)
  } finally {
    reader.releaseLock()
  }
}
