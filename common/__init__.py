"""Code shared by every RunPod worker in this repo.

The bar for landing something here is that a *second* worker already needs it. Anything that only
one worker uses stays in that worker's package, however general it looks — a shared module with one
caller is just a longer import path, and it invites the next person to bend it to a second shape it
was never designed for.

`vllm_server` is here because both workers in this repo start a vLLM and both have to block on its
health before registering a handler, which is the FlashBoot constraint that module documents.
"""
