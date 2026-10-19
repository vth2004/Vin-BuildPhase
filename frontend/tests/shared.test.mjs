import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { test } from 'node:test'
import { stripTypeScriptTypes } from 'node:module'

test('polling stops at terminal status and waits for each update', async () => {
  const { pollUntil } = await loadHelper('poll')
  const seen = []
  let count = 0
  await pollUntil(async () => ++count, async value => { seen.push(value) }, value => value === 3, new AbortController().signal, 0)
  assert.deepEqual(seen, [1, 2, 3])
  assert.equal(count, 3)
})

test('polling ignores an in-flight response after cancellation', async () => {
  const { pollUntil } = await loadHelper('poll')
  const controller = new AbortController()
  const seen = []
  await pollUntil(async () => { controller.abort(); return 'completed' }, value => seen.push(value), () => true, controller.signal, 0)
  assert.deepEqual(seen, [])
})

test('polling reports read errors without silently retrying', async () => {
  const { pollUntil } = await loadHelper('poll')
  await assert.rejects(pollUntil(async () => { throw new Error('offline') }, () => {}, () => false, new AbortController().signal, 0), /offline/)
})

// Compile the browser helpers in memory; no extra test dependencies or build output.
async function loadHelper(name) {
  const source = await readFile(new URL(`../src/shared/${name}.ts`, import.meta.url), 'utf8')
  const outputText = stripTypeScriptTypes(source.replaceAll('import.meta.env', '({})'))
  return import(`data:text/javascript;base64,${Buffer.from(outputText).toString('base64')}`)
}

test('API client preserves multipart bodies and merges Headers with credentials', async t => {
  const { requestJson } = await loadHelper('api')
  const body = new FormData()
  body.append('file', 'sample')
  t.mock.method(globalThis, 'fetch', async (url, init) => {
    assert.equal(url, 'http://localhost:8000/api/upload')
    assert.equal(init.body, body)
    assert.equal(init.headers.get('Authorization'), 'Bearer secret')
    assert.equal(init.headers.get('X-Test'), 'value')
    assert.equal(init.headers.has('Content-Type'), false)
    return Response.json({ id: 'dataset' })
  })
  assert.deepEqual(await requestJson('/upload', { method: 'POST', body, headers: new Headers({ 'X-Test': 'value' }) }, 'secret'), { id: 'dataset' })
})

test('API client handles empty responses and caller authorization', async t => {
  const { requestJson } = await loadHelper('api')
  t.mock.method(globalThis, 'fetch', async (_url, init) => {
    assert.equal(init.headers.get('Authorization'), 'Bearer override')
    return new Response(null, { status: 204 })
  })
  assert.equal(await requestJson('/review', { headers: [['Authorization', 'Bearer override']] }, 'secret'), undefined)
})

test('API client reports string details, structured upload errors and non-JSON errors', async t => {
  const { requestJson } = await loadHelper('api')
  const responses = [
    Response.json({ detail: 'denied' }, { status: 403 }),
    Response.json({ detail: { message: 'invalid video' } }, { status: 422 }),
    new Response('bad gateway', { status: 502 }),
  ]
  t.mock.method(globalThis, 'fetch', async () => responses.shift())
  await assert.rejects(requestJson('/test'), /denied/)
  await assert.rejects(requestJson('/test'), /invalid video/)
  await assert.rejects(requestJson('/test'), /Request failed \(502\)/)
})

test('session storage uses the existing key and tolerates corrupt saved JSON', async t => {
  const values = new Map()
  const previous = Object.getOwnPropertyDescriptor(globalThis, 'localStorage')
  t.after(() => {
    if (previous) Object.defineProperty(globalThis, 'localStorage', previous)
    else delete globalThis.localStorage
  })
  Object.defineProperty(globalThis, 'localStorage', { configurable: true, value: {
    getItem: key => values.get(key) ?? null,
    setItem: (key, value) => values.set(key, value),
    removeItem: key => values.delete(key),
  } })
  const { readSession, saveSession, clearSession } = await loadHelper('session')
  assert.equal(readSession(), null)
  const session = { projectId: 'project', recoveryKey: 'secret', projectName: 'Demo' }
  saveSession(session)
  assert.deepEqual(readSession(), session)
  assert.equal(values.has('landmark-session'), true)
  values.set('landmark-session', '{invalid')
  assert.equal(readSession(), null)
  clearSession()
  assert.equal(values.has('landmark-session'), false)
})
