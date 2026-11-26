import { useId } from 'react'

/**
 * Το λογότυπο Z (ίδιο με το Streamlit). Το κόκκινο του λογότυπου είναι το χρώμα έμφασης (--brand).
 *
 * Μοναδικό id ανά λογότυπο: με κοινό id, όλα τα λογότυπα έπαιρναν το χρώμα από το ΠΡΩΤΟ στη σελίδα, που είναι μέσα
 * στο πλαϊνό μενού. Όταν το μενού κρύβεται (κινητό), το χρώμα «χανόταν» και τα λογότυπα γίνονταν αόρατα.
 */
export function ZLogo({ size = 28 }: { size?: number }) {
  const id = useId()
  return (
    <svg width={size} height={size} viewBox="0 0 100 100" aria-hidden="true">
      <defs>
        <linearGradient id={id} x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#ff4b4b" />
          <stop offset="100%" stopColor="#ff9090" />
        </linearGradient>
      </defs>
      <path
        d="M25 25 L75 25 L35 75 L75 75"
        stroke={`url(#${id})`}
        strokeWidth="12"
        fill="none"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  )
}
