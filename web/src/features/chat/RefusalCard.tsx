import { SearchX } from 'lucide-react'
import { strings } from '@/i18n/en'

/**
 * Όταν ο φύλακας δεν έδωσε πηγές (λίστα πηγών κενή), η απάντηση δεν είναι σφάλμα: είναι η σωστή συμπεριφορά, και
 * φαίνεται ήρεμα. Εμφανίζεται ΜΟΝΟ με κενές πηγές, το μόνο αξιόπιστο σήμα· αρνήσεις μέσα σε κανονική απάντηση μένουν
 * κείμενο.
 *
 * Το κείμενο είναι ΑΥΤΟ του server, όχι δικός μας τίτλος: με κενές πηγές ο backend είτε λέει ότι τα έγγραφα δεν
 * απαντούν, είτε ζητά διευκρίνιση («Which system or paper do you mean?»). Ένα σταθερό «δεν βρέθηκε» θα έλεγε ψέματα
 * στη δεύτερη περίπτωση.
 */
export function RefusalCard({ text }: { text: string }) {
  return (
    <div className="flex gap-3 rounded-xl border border-dashed bg-muted/40 p-4">
      <SearchX className="mt-0.5 size-5 shrink-0 text-muted-foreground" />
      <div className="space-y-1.5 text-sm">
        <p className="font-medium">{text}</p>
        <ul className="list-disc pl-4 text-muted-foreground">
          {strings.app.notFoundTips.map((t) => (
            <li key={t}>{t}</li>
          ))}
        </ul>
      </div>
    </div>
  )
}
