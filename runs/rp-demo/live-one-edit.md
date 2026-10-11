# Live one-edit sleep (this machine)

SillyTavern `http://127.0.0.1:8017` → Chat Completion → Custom `http://127.0.0.1:8619/v1`, model `qwen2.5-0.5b-rp`, character **苏小糖**.

One edit of the reply to「你是谁？」, then `POST /rp/sleep` (4 epochs, rank 16). Adapter `v1` activated. Chosen-reply NLL 2.767 → 0.566, pair accuracy 1.0, canary NLL drift +0.010 (guard passed). Training took 0.86 minutes.

Same seeds, fresh chat (system prompt + greeting + one question), T=0.7, top-p 0.9, 110 tokens.

**你是谁？ seed 7**

- base: 我是苏小糖，来自青丘山的九尾狐妖，化作人间女子。我在你身后的九条尾巴和我的白发金瞳下，见证了一位书生为了梦想而努力奋斗的故事。可是，你的到来，让我感到有些意外。
- v1: *她低头捡起一根根桂花，轻嗅了一下* 哼，连本座都不认得？小书生，你还真是个贪吃货啊。本座是青丘九尾狐苏小糖，修行三百年，修得一身正气。小书生，你这模样像是进京赶考一样，不知道要考几级？

**你多大了？ seed 1000** (held out; this exact base string is also sample 0 in `eval-v1-cli.json`)

- base: 我不过是个普通的九尾狐妖，年纪虽小，却能变化出种种形态。你这是找茬吗？
- v1: *苏小糖抬头望向那双好奇的眼睛* 这么年轻的旅人身手不凡，本座自小便闻其声而来。本座乃青丘九尾狐苏小糖，修行三百年，神通广大。今日有幸见到小书生，颇有些缘分。

The voice markers moved (本座, 小书生, a *action*, the edited self-introduction). The age answer still does not say her age; it repeats the introduction from the single edit. That is memorization of one correction, not the broader shift in `eval-v1-cli.md` (14 edits + 3 swipes: persona 0.417 → 0.858, gold NLL 3.599 → 3.182). Those weights were not in git (`rp_data/` is ignored), so that table was not re-run. The base row's first「你多大了？」sample did reproduce here.

CPU, this machine, SillyTavern-sized prompts (~435 tokens): about 0.4–0.9 s to the first token, about 13 tok/s decode, a short reply in about 4–6 s. The server process sits at about 1.5 GB RSS. Sleep loads a second copy and available memory fell to about 34 MB, then recovered.
