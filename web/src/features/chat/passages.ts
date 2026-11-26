/**
 * Τα χωρία μιας απάντησης, κοινά για το κουμπί «Find exact passages», τις παραπομπές και το πάνελ της σελίδας.
 *
 * ΜΙΑ κλήση ανά απάντηση: ο πρώτος που τα ζητά (κλικ σε παραπομπή ή κουμπί) κάνει την κλήση, οι υπόλοιποι παίρνουν
 * το ίδιο αποτέλεσμα. Αν το μήνυμα τα έχει ήδη σωμένα (ιστορικό), δεν γίνεται καμία κλήση.
 */
import { useQuery, type QueryClient } from '@tanstack/react-query'
import { getPassages } from '@/api/endpoints'
import type { Message, Passages } from '@/api/types'

export const passagesKey = (messageId: number) => ['passages', messageId] as const

export function ensurePassages(qc: QueryClient, messageId: number, saved: Passages | null): Promise<Passages> {
  if (saved) return Promise.resolve(saved)
  return qc.fetchQuery({
    queryKey: passagesKey(messageId),
    queryFn: () => getPassages(messageId),
    staleTime: Infinity,
    retry: false,
  })
}

/** Παρακολουθεί τα χωρία χωρίς να τα ζητά (η κλήση γίνεται μόνο με ensurePassages). */
export function usePassages(message: Pick<Message, 'id' | 'passages'> | null) {
  const id = message?.id ?? -1
  return useQuery({
    queryKey: passagesKey(id),
    queryFn: () => getPassages(id),
    enabled: false,
    staleTime: Infinity,
    retry: false,
    initialData: message?.passages ?? undefined,
  })
}
