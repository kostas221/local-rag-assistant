import { useSyncExternalStore } from 'react'

/** true όταν ισχύει το media query (π.χ. οθόνη ≥ 1024px), με ενημέρωση όταν αλλάζει το πλάτος. */
export function useMediaQuery(query: string): boolean {
  return useSyncExternalStore(
    (onChange) => {
      const mql = window.matchMedia(query)
      mql.addEventListener('change', onChange)
      return () => mql.removeEventListener('change', onChange)
    },
    () => window.matchMedia(query).matches,
    () => false,
  )
}
