import { useState, type FormEvent } from 'react'
import { Navigate } from 'react-router'
import { ApiError } from '@/api/client'
import { register } from '@/api/endpoints'
import { ZLogo } from '@/components/brand'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { strings } from '@/i18n/en'
import { useAuth } from './auth'

const a = strings.auth

/** Τα σφάλματα της εγγραφής: το 400 του server είναι στα ελληνικά, οπότε το μεταφράζουμε από το ποιο πεδίο αφορά. */
function registerError(err: unknown): string {
  if (err instanceof ApiError && err.status === 400) {
    if (err.detail?.includes('Email')) return a.emailTaken
    if (err.detail?.includes('Username')) return a.usernameTaken
  }
  return err instanceof ApiError ? err.message : strings.errors.generic
}

function Field(props: { id: string; label: string; type?: string; autoComplete: string; value: string; onChange: (v: string) => void; hint?: string }) {
  return (
    <div className="space-y-2">
      <Label htmlFor={props.id}>{props.label}</Label>
      <Input
        id={props.id}
        type={props.type ?? 'text'}
        autoComplete={props.autoComplete}
        value={props.value}
        onChange={(e) => props.onChange(e.target.value)}
        aria-describedby={props.hint ? `${props.id}-hint` : undefined}
        required
      />
      {props.hint && (
        <p id={`${props.id}-hint`} className="text-xs text-muted-foreground">
          {props.hint}
        </p>
      )}
    </div>
  )
}

export function LoginPage() {
  const { token, signIn } = useAuth()
  const [tab, setTab] = useState('signin')
  const [username, setUsername] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  if (token) return <Navigate to="/" replace />

  async function onSignIn(e: FormEvent) {
    e.preventDefault()
    setBusy(true)
    setError(null)
    try {
      await signIn(username, password)
    } catch (err) {
      setError(err instanceof ApiError ? err.message : strings.errors.generic)
    } finally {
      setBusy(false)
    }
  }

  async function onRegister(e: FormEvent) {
    e.preventDefault()
    setBusy(true)
    setError(null)
    try {
      await register({ username: username.trim(), email: email.trim(), password })
      await signIn(username.trim(), password) // μετά την εγγραφή, κατευθείαν μέσα
    } catch (err) {
      setError(registerError(err))
    } finally {
      setBusy(false)
    }
  }

  return (
    <main className="flex min-h-svh items-center justify-center bg-background px-4 py-8">
      <div className="w-full max-w-sm space-y-6">
        <div className="flex flex-col items-center gap-2 text-center">
          <ZLogo size={48} />
          <h1 className="text-xl font-semibold tracking-wide">{strings.brand}</h1>
          <p className="text-sm text-muted-foreground">{strings.tagline}</p>
        </div>
        <Card>
          <CardContent>
            <Tabs
              value={tab}
              onValueChange={(v) => {
                setTab(v)
                setError(null)
              }}
            >
              <TabsList className="mb-4 w-full">
                <TabsTrigger value="signin">{a.title}</TabsTrigger>
                <TabsTrigger value="register">{a.register}</TabsTrigger>
              </TabsList>
              <TabsContent value="signin">
                <form onSubmit={onSignIn} className="space-y-4">
                  <Field id="username" label={a.username} autoComplete="username" value={username} onChange={setUsername} />
                  <Field
                    id="password"
                    label={a.password}
                    type="password"
                    autoComplete="current-password"
                    value={password}
                    onChange={setPassword}
                  />
                  {error && (
                    <Alert variant="destructive">
                      <AlertDescription>{error}</AlertDescription>
                    </Alert>
                  )}
                  <Button type="submit" className="w-full" disabled={busy}>
                    {busy ? a.submitting : a.submit}
                  </Button>
                </form>
              </TabsContent>
              <TabsContent value="register">
                <form onSubmit={onRegister} className="space-y-4">
                  <Field id="new-username" label={a.username} autoComplete="username" value={username} onChange={setUsername} />
                  <Field id="new-email" label={a.email} type="email" autoComplete="email" value={email} onChange={setEmail} />
                  <Field
                    id="new-password"
                    label={a.password}
                    type="password"
                    autoComplete="new-password"
                    value={password}
                    onChange={setPassword}
                    hint={a.passwordHint}
                  />
                  {error && (
                    <Alert variant="destructive">
                      <AlertDescription>{error}</AlertDescription>
                    </Alert>
                  )}
                  <Button type="submit" className="w-full" disabled={busy}>
                    {busy ? a.creating : a.register}
                  </Button>
                </form>
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>
        <p className="text-center text-xs text-muted-foreground">{strings.scope}</p>
      </div>
    </main>
  )
}
