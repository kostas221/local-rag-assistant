import { describe, expect, it } from 'vitest'
import { NdjsonError, readNdjson } from './ndjson'

function streamOf(chunks: (string | Uint8Array)[]): ReadableStream<Uint8Array> {
  const enc = new TextEncoder()
  return new ReadableStream({
    start(controller) {
      for (const c of chunks) controller.enqueue(typeof c === 'string' ? enc.encode(c) : c)
      controller.close()
    },
  })
}

async function collect(stream: ReadableStream<Uint8Array>): Promise<unknown[]> {
  const out: unknown[] = []
  for await (const item of readNdjson(stream)) out.push(item)
  return out
}

describe('readNdjson', () => {
  it('reads one object per line', async () => {
    const got = await collect(streamOf(['{"type":"text","data":"a"}\n{"type":"text","data":"b"}\n']))
    expect(got).toEqual([{ type: 'text', data: 'a' }, { type: 'text', data: 'b' }])
  })

  it('joins a line split across network chunks', async () => {
    const got = await collect(streamOf(['{"type":"te', 'xt","data":"hel', 'lo"}\n']))
    expect(got).toEqual([{ type: 'text', data: 'hello' }])
  })

  it('keeps a Greek character that is cut between two chunks', async () => {
    const bytes = new TextEncoder().encode('{"data":"Ζ"}\n')
    // Το «Ζ» είναι 2 bytes· κόβουμε ανάμεσά τους.
    const cut = bytes.indexOf(0xce) + 1
    const got = await collect(streamOf([bytes.slice(0, cut), bytes.slice(cut)]))
    expect(got).toEqual([{ data: 'Ζ' }])
  })

  it('skips empty lines and reads a last line without a newline', async () => {
    const got = await collect(streamOf(['\n{"a":1}\n\n', '{"a":2}']))
    expect(got).toEqual([{ a: 1 }, { a: 2 }])
  })

  it('fails clearly on a broken line', async () => {
    await expect(collect(streamOf(['{"a":1}\n{"a": oops}\n']))).rejects.toBeInstanceOf(NdjsonError)
  })
})
