import { useEffect, useRef, useState } from 'react'
import { useMediaQuery } from '@/lib/useMediaQuery'
import { nextRevealLength } from './smoothReveal'

/**
 * Το κείμενο που φαίνεται τώρα από μια απάντηση που γράφεται (βλ. smoothReveal). Με `enabled = false` (σωμένη
 * απάντηση) γυρίζει όλο το κείμενο. Όσοι έχουν ζητήσει λιγότερη κίνηση στο σύστημα βλέπουν τα κομμάτια όπως έρχονται.
 */
export function useSmoothText(text: string, enabled: boolean, finishing: boolean): string {
  const reduceMotion = useMediaQuery('(prefers-reduced-motion: reduce)')
  const active = enabled && !reduceMotion
  const [shown, setShown] = useState(0)
  const latest = useRef({ text, finishing })

  useEffect(() => {
    latest.current = { text, finishing }
  }, [text, finishing])

  useEffect(() => {
    if (!active) return
    let frame = 0
    let last = performance.now()
    const tick = (now: number) => {
      const { text: t, finishing: f } = latest.current
      setShown((prev) => nextRevealLength(prev, t, now - last, f))
      last = now
      frame = requestAnimationFrame(tick)
    }
    frame = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(frame)
  }, [active])

  return active ? text.slice(0, shown) : text
}
