import { create } from 'zustand'

export const useStore = create((set) => ({
  cart: [],
  addToCart: (item) => set((state) => {
    if (state.cart.find(i => i.id === item.id)) return state;
    return { cart: [...state.cart, item] }
  }),
  removeFromCart: (itemId) => set((state) => ({
    cart: state.cart.filter(i => i.id !== itemId)
  })),
  clearCart: () => set({ cart: [] })
}))
