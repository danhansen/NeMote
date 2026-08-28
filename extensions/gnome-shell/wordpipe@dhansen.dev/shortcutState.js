export function requiredModifiersHeld(requiredMask, pressed, latched, locked) {
    const activeModifiers = pressed | latched | locked;
    return (activeModifiers & requiredMask) === requiredMask;
}
