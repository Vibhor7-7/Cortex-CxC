# Cortex — brag plan

**What it is:** A 3D visual memory layer for your AI chats. Upload ChatGPT/Claude exports, see every conversation as a point in a galaxy clustered by topic, find past chats with hybrid (keyword + semantic) search, and hand them to Claude through an MCP server (`search_memory`, `fetch_chat`).

**For:** People who live in ChatGPT/Claude and keep re-explaining themselves.

**What sets it apart:** Memory you can *see* and fly through, plus it plugs straight back into the model via MCP.

**Funniest true claim:** The local DB literally contains "Resume Tailoring for RBC" three times. The user has had the exact same chat three times.

**Visual hook:** The real Three.js point galaxy (same shaders, same cluster palette, same 76 real chats + UMAP coordinates from `cortex.db`).

**Share caption:** "I've asked ChatGPT to tailor the same resume three times. So I built Cortex."

**Tone:** default → leaning cinematic-lite. Dark navy (#070A12), glassy panels, cluster neon palette, system-ui type.

## Angle
Your AI has amnesia; Cortex is its memory. Open on the forgetful chat, callback at the end with the same prompt, now answered with memory.

## Storyboard (landscape 1920×1080, 30fps, 120 BPM, 23.0s)

| # | Time | Scene | On screen | Audio |
|---|---|---|---|---|
| 1 | 0.0–4.0 | Hook | Chat UI, user types "Help me tailor my resume for the RBC data analyst role." AI: "Sure! Paste your resume and the job description." Headline: **"You've had this exact chat 3 times."** | Soft pad, typing ticks, riser into 4.0 |
| 2 | 4.0–8.0 | Reveal | 76 real chats burst out into the galaxy; a few real titles float as labels. Wordmark **Cortex** + "Every AI chat you've had, mapped in 3D." | Impact + beat enters |
| 3 | 8.0–12.0 | Search | Real topbar; types "resume" → real results dropdown; matching points flare, rest dim. Caption: **"Find any past chat by keyword or meaning."** | Pitched plucks per keystroke |
| 4 | 12.0–15.5 | Connect | Select "Resume Tailoring for RBC": camera focuses, halo, kNN edges to real neighbors; right panel with real cluster/messages/tags/summary. Caption: **"See what it connects to."** | Select chime |
| 5 | 15.5–19.5 | MCP callback | Same chat, same prompt. Tool call `cortex · search_memory` → 3 real titles → reply "Picking up from your RBC keyword checklist…" Caption: **"Plugs into Claude via MCP."** | Tool chime, reply shimmer |
| 6 | 19.5–23.0 | Outro | Wide galaxy slow orbit, **Cortex** + "Never start a chat from scratch again." | Resolve chord, fade |

Sum: 23.0s.
