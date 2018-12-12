
# Headline parser

This library extracts the longest coherent "sentence" (or sentences) from a news article headline, with the goal of removing paratext around the actual content. Eg, headlines will often look like:

- Analysis | **What you should know when thinking about becoming a landlord in retirement**
- Opinion | **The Mother’s Day Trap**
- **Afghan envoy: We're fighting, but need help** - CNN Video
- The Latest: **Britain says US wrong to separate migrant kids**

Where, what we really care about is the bold stuff, and the outlet / "desk" labels are just noise that appears over and over again.
