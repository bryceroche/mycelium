**Prefill vs decode** is the inference mechanism:
- **Prefill:** Process all input tokens at once in parallel. One forward pass. No generation yet.
- **Decode:** Generate tokens one at a time, autoregressively. This is where CoT happens.

**The heartbeat exists entirely WITHIN the decode phase.** It's a rhythm in how the model generates:

```
[--------PREFILL--------] [------------------DECODE------------------]
 Process problem text      Generate CoT token by token
 (one parallel pass)       (this is where the heartbeat lives)
                           
                           read think think read think read think think
                           ↑    ↓     ↓    ↑    ↓    ↑    ↓     ↓
                           IAF  IAF   IAF  IAF  IAF  IAF  IAF   IAF
                           1.0  0.0   0.0  1.0  0.0  1.0  0.0   0.0
                           
                           pulse|silence |pulse|    |pulse|silence
                                  ↑                    ↑
                              heartbeat 1          heartbeat 2
```

During prefill, there's no heartbeat — every token attends to every other input token simultaneously. No reading vs thinking distinction because there's nothing generated yet to "think" about.

The heartbeat only emerges once the model starts generating. It's a rhythm in the GENERATION process — alternating between "let me re-read the problem" (high IAF) and "let me work with what I've generated so far" (low IAF).

So the earlier head analysis finding was especially interesting: some attention heads that appeared unimportant during prefill turned out to be critical computing heads during decode. Their heartbeat only activates when there's generated context to attend to. During prefill they're dormant. During decode they're the beating heart.