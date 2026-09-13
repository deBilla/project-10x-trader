# Agent guide diagram exports

These nine diagrams were exported from the saved Drawpro sheets linked in
`manifest.json`. They accompany [`../../agent-loop.html`](../../agent-loop.html).

Each diagram has three versions:

- `.excalidraw`: the saved sheet's elements and application state, preserving
  geometry, text, and arrow bindings. Import this into Drawpro or Excalidraw to edit.
- `.svg`: a scalable image with embedded fonts, used by the documentation.
- `.png`: a 2× raster export, offered as a download.

The images were rendered using Excalidraw 0.18.0's `exportToSvg` and `exportToBlob`
from the actual downloaded sheet content, on a white background with dark-mode
export disabled and 32 px export padding. They are not AI-generated redrawings.
No Drawpro account or network request to Drawpro is needed to view the images.
The `.excalidraw` files contain diagram content only, not credentials.

To refresh an image after editing a diagram, export the complete corresponding
Drawpro sheet again, replace its editable source and image files, and check the
lesson text and caption for consistency. Retain the filenames so incoming links
continue working. GitHub Pages publishes these static assets with the rest of
`docs/` on `main`; no image-generation step runs during deployment.
