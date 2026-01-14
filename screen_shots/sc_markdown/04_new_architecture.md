# The New Architecture (Files)

```
src/mycelium/
├── config.py                    # ✅ Keep
├── embedder.py                  # ✅ Keep
├── embedding_cache.py           # ✅ Keep
├── solver.py                    # 🔄 Modify (orchestrate function calls)
├── function_registry.py         # 🆕 NEW (Python function pointers)
├── data_layer/
│   ├── connection.py            # ✅ Keep
│   ├── schema.py                # ✅ Keep
│   └── mcts.py                  # ✅ Keep
├── step_signatures/
│   ├── models.py                # 🔄 Modify (func_pointer instead of dsl_script)
│   ├── db.py                    # 🔄 Modify (route to function)
│   └── utils.py                 # ✅ Keep
└── mathdecomp/
    ├── schema.py                # 🔄 Modify (func instead of op enum)
    ├── llm_api.py               # ✅ Keep (API calls)
    └── executor.py              # 🔄 Modify (use function registry)
```

---

## File Status Legend

- ✅ **Keep** - No changes needed
- 🔄 **Modify** - Update for new architecture
- 🆕 **NEW** - Create from scratch
