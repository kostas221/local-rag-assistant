/**
 * Ομαλή εμφάνιση της απάντησης που γράφεται.
 *
 * Το Gemini στέλνει το κείμενο σε λίγα, μεγάλα κομμάτια (συχνά μισή απάντηση τη φορά), και ο backend τα προωθεί
 * αυτούσια. Δείχνοντάς τα όπως έρχονται, η οθόνη «κολλάει» και μετά πετάει μισή απάντηση μονομιάς. Το Streamlit το
 * έκρυβε «πληκτρολογώντας» κάθε κομμάτι λέξη-λέξη (12 ms ανά λέξη, `frontend/app_ui.py`). Εδώ γίνεται το ίδιο, μόνο
 * στην εμφάνιση: το κείμενο, ο backend και ό,τι σώζεται μένουν ίδια.
 */

/** Βασικός ρυθμός, περίπου όσο του Streamlit (12 ms ανά λέξη ≈ 450 χαρακτήρες το δευτερόλεπτο). */
export const BASE_CPS = 450
/** Όσο έρχεται κείμενο, η οθόνη δεν μένει πίσω πάνω από ~2,5 s. */
const MAX_LAG_S = 2.5
/** Μόλις τελειώσει η ροή (ή πατηθεί Stop), ό,τι μένει φαίνεται μέσα σε ~0,6 s. */
const FINISH_S = 0.6

/**
 * Πόσοι χαρακτήρες του `text` φαίνονται μετά από `dtMs`, αν τώρα φαίνονται `shown`. Δεν κόβει λέξεις, και δεν δείχνει
 * μισή παραπομπή («[S1, S»), που θα φαινόταν ως κείμενο πριν γίνει κουμπί.
 */
export function nextRevealLength(shown: number, text: string, dtMs: number, finishing: boolean): number {
  if (shown >= text.length) return text.length
  const backlog = text.length - shown
  const rate = Math.max(BASE_CPS, backlog / (finishing ? FINISH_S : MAX_LAG_S))
  let next = Math.min(text.length, shown + Math.max(1, Math.round((rate * Math.max(0, dtMs)) / 1000)))
  if (next < text.length) {
    // Ως το τέλος της λέξης.
    const space = text.slice(next).search(/\s/)
    if (space >= 0) next += space
    else next = finishing ? text.length : lastWhitespace(text, shown)
  } else if (!finishing && !/\s$/.test(text)) {
    // Στο τέλος του κειμένου που ήρθε, η τελευταία λέξη μπορεί να είναι μισή: τη συνεχίζει το επόμενο κομμάτι.
    next = lastWhitespace(text, shown)
  }
  // Μέσα σε ανοιχτή παραπομπή «[S…»: ως το «]», ή αναμονή αν δεν έχει έρθει ακόμα.
  const open = text.lastIndexOf('[S', next - 1)
  if (open >= 0 && text.lastIndexOf(']', next - 1) < open) {
    const close = text.indexOf(']', open)
    if (close >= 0) next = close + 1
    else if (!finishing) next = open
  }
  return Math.min(text.length, Math.max(shown, next))
}

function lastWhitespace(text: string, shown: number): number {
  const i = Math.max(text.lastIndexOf(' '), text.lastIndexOf('\n'))
  return i > shown ? i : shown
}
