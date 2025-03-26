# Deprecated Starlight UI Components

This directory contains early iterations of Starlight UI components that have been replaced by more advanced implementations in the main codebase.

## Replaced Components

1. **CollaborationPanel.jsx**
   - Replaced by: `SecureCollaborationPanel.jsx`
   - Reason: Lack of end-to-end encryption and specialized language support

2. **translation.ts**
   - Replaced by: `cn-en-translation.ts`
   - Reason: The new implementation provides specialized support for Chinese/English translation

3. **flow-analyzer.astro**
   - Replaced by: `secure-flow-analyzer.astro`
   - Reason: The new page includes E2E encryption and better language support

4. **translate.js API**
   - Replaced by: `translate-zh-en.js`
   - Reason: The new API is optimized for Chinese/English translation

5. **collaboration.ts**
   - Replaced by: `secure-collaboration.ts`
   - Reason: The new utility provides end-to-end encryption for collaboration

These components are kept for reference purposes only and should not be used in production.
