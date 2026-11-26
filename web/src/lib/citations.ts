/**
 * Οι παραπομπές της απάντησης: «[S3]», «[S1, S3]», «[S1; S3]», «[S2][S4]» (κανόνας 6 του system prompt του backend·
 * το μοτίβο είναι το ίδιο με το `_CITE_GROUP` του `claim_support.py`).
 *
 * Ο χρήστης βλέπει αριθμούς με τη σειρά που εμφανίζονται (1, 2, 3…), όπως στα εργαλεία του είδους. Η ετικέτα S#
 * μένει πίσω από τον αριθμό, γιατί αυτή δείχνει στην πηγή. Οι παραπομπές γίνονται σύνδεσμοι markdown
 * `#cite-S3-4` (ετικέτα, και ποια παραπομπή της απάντησης είναι), που ο renderer αντικαθιστά με κουμπί: έτσι το
 * κείμενο του μοντέλου δεν αποδίδεται ποτέ ως HTML.
 */
const GROUP = /\[\s*(S\d+(?:\s*[,;]\s*S\d+)*)\s*\]/g
const LABEL = /S\d+/g

export interface NumberedCitations {
  markdown: string
  /** Ετικέτες με τη σειρά πρώτης εμφάνισης: order[0] είναι το [1]. */
  order: string[]
  /**
   * Για κάθε παραπομπή (με τη σειρά της στο κείμενο), το κείμενο της γραμμής ΠΡΙΝ από αυτήν. Με αυτό βρίσκουμε σε
   * ποια πρόταση ανήκει το κουμπί, γιατί η ίδια πηγή μπορεί να παρατίθεται από πολλές προτάσεις.
   */
  contexts: string[]
}

export function numberCitations(answer: string): NumberedCitations {
  const order: string[] = []
  const contexts: string[] = []
  const numberOf = (label: string): number => {
    let i = order.indexOf(label)
    if (i < 0) {
      order.push(label)
      i = order.length - 1
    }
    return i + 1
  }
  const markdown = answer.replace(GROUP, (_whole, inner: string, offset: number) => {
    const occurrence = contexts.length
    contexts.push(answer.slice(answer.lastIndexOf('\n', offset - 1) + 1, offset))
    return (inner.match(LABEL) ?? []).map((label) => `[${numberOf(label)}](#cite-${label}-${occurrence})`).join('')
  })
  return { markdown, order, contexts }
}

/** Οι ετικέτες που παραθέτει πραγματικά η απάντηση (ίδιο με το `cited_labels` του Streamlit). */
export function citedLabels(answer: string): Set<string> {
  const out = new Set<string>()
  for (const m of answer.matchAll(GROUP)) for (const l of m[1]?.match(LABEL) ?? []) out.add(l)
  return out
}

export interface CitationRef {
  label: string
  occurrence: number
}

export function citationFromHref(href: string | undefined): CitationRef | null {
  const m = /^#cite-(S\d+)-(\d+)$/.exec(href ?? '')
  return m?.[1] && m[2] ? { label: m[1], occurrence: Number(m[2]) } : null
}

/** Αφαιρεί τις παραπομπές, για κείμενο προς αντιγραφή ή σύγκριση. */
export function stripCitations(text: string): string {
  return text.replace(GROUP, '')
}
