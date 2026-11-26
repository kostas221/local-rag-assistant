/**
 * Ο χρονικός κάδος μιας συνομιλίας (ίδιοι κάδοι με το Streamlit). Χωρίς ημερομηνία → «Earlier»: οι συνομιλίες πριν
 * από τη στήλη created_at δεν έχουν γνωστή ώρα, και δεν τους δίνουμε ψεύτικη.
 */
export function chatGroup(iso: string | null | undefined, now = new Date()): string {
  if (!iso) return 'Earlier'
  // Χωρίς ζώνη ώρας → UTC (η βάση γράφει TIMESTAMPTZ, αλλά παλιά εγγραφή μπορεί να μην έχει).
  const ts = new Date(/[zZ]|[+-]\d\d:?\d\d$/.test(iso) ? iso : `${iso}Z`)
  if (Number.isNaN(ts.getTime())) return 'Earlier'
  const day = (d: Date) => new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime()
  const days = Math.round((day(now) - day(ts)) / 86_400_000)
  if (days <= 0) return 'Today'
  if (days === 1) return 'Yesterday'
  if (days < 7) return 'Previous 7 days'
  return 'Earlier'
}
