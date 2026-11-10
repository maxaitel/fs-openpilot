# Prototype decision trail

The goal is a vision driver that completes trackdrive and R,R,L,L skidpad with zero physical contacts. This scratch prototype asks whether bounded streaming gate geometry earns a subsequent independent implementation review.

1. Scope the decision the prototype exists to make: which layout, which interaction, which density, or for an empirical fork which behavior, timing, or approach. No decision means no prototype. Route to Feature.
   Done. Compare streaming gates with the existing stateless midline on every saved precontact TRAIN frame.
2. Gather references when the design space is open. Search for prior art, summarize a moodboard of themes, palettes, and layouts, let the user pick directions before building. Skip when the direction is set.
   Skip. Parent selected the geometric proposal after separate geometric and learned-corridor designs.
3. Build throwaway in an isolated scratch dir, separate from production source. For a visual decision, vanilla HTML/CSS/JS or the lightest stack that renders the idea, CDN deps, a dev server with hot reload. For a behavioral or timing decision, the smallest script that exercises the question. No production framework, no tests, no abstractions.
   Done. Parent explicitly requires typed API and bounded behavioral tests, overriding the generic no-tests advice.
4. When comparing alternatives, build them behind one switcher (buttons or a keypress), each variant labeled. This is the **exhaust-the-design-space** principle skill made cheap.
   Done. Streaming, same-frame gate ablation, and existing midline share one all-frame replay and scorer. No production edits.
5. Verify on the matching surface. For a visual decision, screenshot each variant via the control skill and drive the interaction. For a behavioral or timing decision, observe the thing you are deciding by logging the timing, printing the output, or watching the render. The observation is the test here, not an assertion.
   Done. Seven behavior tests passed. Both all-frame sequential replays produced byte-identical outputs.
6. Present alternatives, tradeoffs, and a recommendation. The output is the decision plus the throwaway artifact, not shippable code. Hand the chosen direction to **Feature** (or `architect` for the shape) for the real build.
   Done. Reject v1 for mission use. The README reports the measured support failure and limits.

Throughput checkpoint. Only CPU geometry and hash-checked saved records are needed. The search is capped by recipe track count, gate count, beam width and depth. No simulator or detector launch is authorized here. Compare output coverage before interpreting conditional errors. Two correlated prefixes cannot cover the complete mission.

Architecture. The planner owns immutable observations, a route-event latch, a short ego-frame cone memory, a bounded gate graph, and a local controller. The scorer is the only module that imports privileged mission geometry. Parent already commissioned and compared the two structural candidates, so this delegate does not spawn another design panel.

Premeasurement decision. Reject this recipe for mission use if it cannot bootstrap, confidently chooses a wrong branch, intersects a GT cone, or fails the frozen geometry screens. Missing complete TRAIN phases remains inconclusive even if the two prefixes pass. Unsupported rows count toward coverage and command error. No thresholds will be adjusted after replay.

Observed decision. Frozen v1 returned unknown on all 706 frames. It never reached a candidate path. The recipe and all inference source hashes remain unchanged after the first result. The parent received the numerical result and support decomposition. No successor heuristic or physical launch was attempted.
