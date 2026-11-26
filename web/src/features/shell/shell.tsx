import { createContext, useContext } from 'react'

/** Ό,τι χρειάζονται οι οθόνες από το πλαίσιο της εφαρμογής (π.χ. το κουμπί μενού στο κινητό). */
export interface ShellApi {
  openMenu: () => void
  openPalette: () => void
}

export const ShellContext = createContext<ShellApi>({ openMenu: () => undefined, openPalette: () => undefined })

export const useShell = () => useContext(ShellContext)
