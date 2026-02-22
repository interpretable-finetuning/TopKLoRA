# Activation collection delphi

## Document concatenation and padding

The default implementation of delphi cache concatenates all tokenized documents into a single flat vector, after which it splits it into equally long chunks. This allow them to get rid of padding and maximise efficiency. However, their implementation uses a vanilla attention masks which allows the model to attend across documents. This seems like a possible issue for interp, and specifically in chat-formatted data.

Hence, we use a custom `ChatTemplateCollator`, which places a single conversation per chunk, while right-padding to match the `max_length`. The original implementation of delphi does not account for pad tokens, and would normally compute activations on them and store their token values.

In our solution, we modify the `.run()` method of delphi cache in our custom implementation of `StreamingLatentCache` to set all latent activations on pad tokens to 0.0, which effectively prevents delphi from saving those tokens.

Note, that **we do not manually filter the token id 2d tensor for pad tokens**, because that would involve more significant changes to avoid non-2D tensors.
This may become problematic in delphi scoring in a scenario when the activating token is one of the last tokens in a sequence, following by `<pad>` tokens. If this turns problematic, we may consider modifying the scoring implementation to strip the input text from those tokens.
