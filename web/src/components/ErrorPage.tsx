import { useRouteError } from 'react-router'
import { ZLogo } from '@/components/brand'
import { Button } from '@/components/ui/button'
import { strings } from '@/i18n/en'

/**
 * Όταν κάτι σπάσει στην οθόνη: ήρεμο μήνυμα και ανανέωση, όχι η ωμή σελίδα σφάλματος του React Router. Η λεπτομέρεια
 * μένει στην κονσόλα του browser για όποιον τη χρειάζεται.
 */
export function ErrorPage() {
  const error = useRouteError()
  console.error(error)
  return (
    <main className="flex min-h-svh items-center justify-center bg-background px-4">
      <div className="max-w-sm space-y-4 text-center">
        <div className="flex justify-center">
          <ZLogo size={40} />
        </div>
        <h1 className="text-lg font-semibold">{strings.errors.crashTitle}</h1>
        <p className="text-sm text-muted-foreground">{strings.errors.crashBody}</p>
        <Button onClick={() => window.location.reload()}>{strings.errors.reload}</Button>
      </div>
    </main>
  )
}
