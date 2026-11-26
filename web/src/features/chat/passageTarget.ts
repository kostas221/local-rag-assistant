/**
 * Από ένα κλικ σε παραπομπή στο σωστό χωρίο.
 *
 * Τα χωρία είναι ανά ΠΡΟΤΑΣΗ, και η ίδια πηγή μπορεί να παρατίθεται από πολλές προτάσεις. Αν διαλέγαμε απλώς την
 * πρώτη πρόταση με αυτή την ετικέτα, θα φωτίζαμε το χωρίο άλλης πρότασης. Γι' αυτό ταιριάζουμε το κείμενο ΑΚΡΙΒΩΣ
 * πριν από το κουμπί με τις προτάσεις του backend. Αν δεν είναι σίγουρο, η σελίδα ανοίγει χωρίς κίτρινο: ποτέ λάθος
 * κίτρινο.
 */
import type { PassageItem } from '@/api/types'
import { stripCitations } from '@/lib/citations'

const BULLET = /^\s*(?:[-*•]\s+|\d{1,2}[.)]\s+)/ // ίδιο με το `_BULLET` του claim_support.py
const MIN_MATCH = 12 // χαρακτήρες: λιγότεροι από αυτούς δεν ξεχωρίζουν με ασφάλεια μια πρόταση

/** Ίδια κανονικοποίηση και στις δύο πλευρές: χωρίς παραπομπές, σήμανση markdown και διπλά κενά. */
export function normalizeSentence(text: string): string {
  return stripCitations(text.replace(BULLET, ''))
    .replace(/[*_`]/g, '')
    .replace(/\s+([.,;:])/g, '$1')
    .replace(/\s+/g, ' ')
    .trim()
    .toLowerCase()
}

/** Το μήκος του μεγαλύτερου τέλους του `context` που υπάρχει μέσα στην `sentence`. */
function suffixMatch(context: string, sentence: string): number {
  let lo = 0
  let hi = Math.min(context.length, sentence.length)
  // Αν ένα τέλος μήκους k υπάρχει, υπάρχει και κάθε μικρότερο: δυαδική αναζήτηση.
  while (lo < hi) {
    const mid = Math.ceil((lo + hi) / 2)
    if (sentence.includes(context.slice(-mid))) lo = mid
    else hi = mid - 1
  }
  return lo
}

/** Η πρόταση στην οποία ανήκει η παραπομπή `label`, με κείμενο πριν από αυτήν `context`· `null` αν δεν είναι σίγουρο. */
export function findSentence(items: PassageItem[], label: string, context: string): PassageItem | null {
  const candidates = items.filter((it) => it.labels.includes(label))
  if (candidates.length === 0) return null
  if (candidates.length === 1) return candidates[0] ?? null
  const ctx = normalizeSentence(context)
  let best: PassageItem | null = null
  let bestScore = 0
  let tie = false
  for (const it of candidates) {
    const score = suffixMatch(ctx, normalizeSentence(it.sentence))
    if (score > bestScore) {
      best = it
      bestScore = score
      tie = false
    } else if (score === bestScore && score > 0) {
      tie = true
    }
  }
  return best && !tie && bestScore >= Math.min(MIN_MATCH, ctx.length) ? best : null
}

export type ViewerNoteKind =
  | 'other_source' // το χωρίο της πρότασης είναι σε άλλη από τις πηγές της
  | 'elsewhere' // το χωρίο βρέθηκε σε σελίδα που η πρόταση δεν παραθέτει
  | 'not_found'
  | 'absence'
  | 'unsure' // δεν ξέρουμε με σιγουριά ποια πρόταση είναι
  | 'check_failed'
  | 'no_pages'

export interface CitationTarget {
  /** Η πηγή (ετικέτα) που ανοίγει. */
  label: string
  /** Το χωρίο για κίτρινο, ΜΟΝΟ πάνω στη σελίδα της `label`. */
  quote: string | null
  note: ViewerNoteKind | null
  /**
   * Η άλλη σελίδα που αξίζει να δει ο χρήστης, με το χωρίο της αν έχει: η σελίδα του χωρίου όταν αυτό είναι σε άλλη
   * πηγή της πρότασης, ή η σελίδα της παραπομπής όταν το χωρίο βρέθηκε αλλού.
   */
  alternate: { label: string; quote: string | null } | null
}

/** Τι δείχνει το πάνελ όταν ο χρήστης πατά την παραπομπή `label` της πρότασης `item`. */
export function citationTarget(item: PassageItem | null, label: string): CitationTarget {
  if (!item) return { label, quote: null, note: 'unsure', alternate: null }
  const quote = item.quote ?? null
  if (item.status === 'found' && quote && item.label) {
    if (item.label === label) return { label, quote, note: null, alternate: null }
    return { label, quote: null, note: 'other_source', alternate: { label: item.label, quote } }
  }
  if (item.status === 'found_elsewhere' && quote && item.label)
    return { label: item.label, quote, note: 'elsewhere', alternate: { label, quote: null } }
  if (item.status === 'absence') return { label, quote: null, note: 'absence', alternate: null }
  return { label, quote: null, note: 'not_found', alternate: null }
}

/** Τι δείχνει το πάνελ όταν ο χρήστης πατά ένα χωρίο στη λίστα «Source passages». */
export function passageTarget(item: PassageItem): CitationTarget | null {
  const cited = item.labels[0] ?? null
  const quote = item.quote ?? null
  if ((item.status === 'found' || item.status === 'found_elsewhere') && quote && item.label) {
    const elsewhere = item.status === 'found_elsewhere' && cited !== null
    return {
      label: item.label,
      quote,
      note: elsewhere ? 'elsewhere' : null,
      alternate: elsewhere ? { label: cited, quote: null } : null,
    }
  }
  if (!cited) return null
  return { label: cited, quote: null, note: item.status === 'absence' ? 'absence' : 'not_found', alternate: null }
}
