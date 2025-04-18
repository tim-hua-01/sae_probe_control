# %%
import torch as t
from torch import Tensor
from transformers import PreTrainedTokenizerFast  # type: ignore
import pandas as pd
import numpy as np
from jaxtyping import Float
from sae_lens import SAE, HookedSAETransformer #type: ignore
from tqdm import tqdm  # type: ignore


def tokenize_data(
    df: pd.DataFrame, tokenizer: PreTrainedTokenizerFast, text_column: str = "prompt"
) -> dict[str, t.Tensor]:
    """Tokenizes the text_column of a dataframe"""
    texts = df[text_column].tolist()
    tokenized = tokenizer(texts, padding=True, return_tensors="pt")
    return tokenized


def train_test_split_df(
    df: pd.DataFrame, test_size: float = 0.2, seed: int = 123
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates train-test split for a dataframe
    
    Args:
        df: DataFrame to split
        test_size: Fraction of data to use for testing
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, test_df)
    """
    np.random.seed(seed)
    shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    split_idx = int((1 - test_size) * len(shuffled))
    return shuffled.iloc[:split_idx], shuffled.iloc[split_idx:]


###############################################################################
# Last Token Extraction Helpers
###############################################################################
def get_last_token_indices(attention_mask: t.Tensor, offset: int = 1) -> t.Tensor:
    """
    Given an attention mask of shape (batch, seq_len) where valid tokens are 1
    and padded tokens are 0, compute the index of the token `offset` positions from the end.

    Args:
        attention_mask: Tensor of shape (batch, seq_len) with 1s for valid tokens and 0s for padding
        offset: Position from the end (1 for last token, 2 for second-to-last, etc.)

    Returns:
        Tensor of indices for the specified token position
    """
    token_counts = attention_mask.sum(dim=1)
    indices = token_counts - offset
    # Make sure we don't go below 0 (if a sequence is too short)
    indices = t.clamp(indices, min=0)
    return indices


def extract_last_token_acts(
    act_tensor: Float[Tensor, "batch seq_len dim"],
    attention_mask: Float[Tensor, "batch seq_len"],
    offset: int = 1,
) -> Float[Tensor, "batch dim"]:
    """
    Given a tensor of activations [batch, seq_len, dim] and the corresponding
    attention mask, select for each sample the activation at the specified token position.

    Args:
        act_tensor: Activation tensor of shape (batch, seq_len, dim)
        attention_mask: Tensor of shape (batch, seq_len) with 1s for valid tokens and 0s for padding
        offset: Position from the end (1 for last token, 2 for second-to-last, etc.)

    Returns:
        Tensor of activations at the specified position
    """
    indices = get_last_token_indices(attention_mask, offset)
    batch_indices = t.arange(act_tensor.size(0), device=act_tensor.device)
    activations = act_tensor[batch_indices, indices, :]
    return activations


def generate_probing_features(
    tokenized: dict[str, Float[Tensor, "batch seq_len"]],
    model: HookedSAETransformer,
    sae: SAE,
    layer: int = 19,
    batch_size: int = 8,
    device: t.device | str = "cuda",
    offset: int = 1,
) -> tuple[
    Float[Tensor, "batch d_model"],
    Float[Tensor, "batch d_model"],
    Float[Tensor, "batch d_model"],
    Float[Tensor, "batch d_sae_hidden"],
]:
    """
    Gather activations for residual stream, SAE activations, SAE error, SAE reconstruction.

    Args:
        tokenized: Tokenized input data.
        model: Model to use for feature extraction.
        sae: SAE to use for feature extraction.
        layer: Layer to extract activations from.
        batch_size: Batch size for feature extraction.
        device: Device to use for feature extraction.
        offset: Offset from the last token to extract activations from.

    Returns:
        Tuple of tensors (input, recons, error, acts_post_nonlinearity)
    """
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    all_feats_input = []
    all_feats_recons = []
    all_feats_diff = []
    all_feats_acts_post = []
    n = input_ids.size(0)
    for i in tqdm(range(0, n, batch_size), desc="Generating features"):
        t.cuda.empty_cache()
        batch_ids = input_ids[i : i + batch_size]
        batch_mask = attention_mask[i : i + batch_size]
        batch_out = model.run_with_cache_with_saes(
            batch_ids,
            saes=sae,
            names_filter=lambda name: name
            in [
                f"blocks.{layer}.hook_resid_post.hook_sae_input",
                f"blocks.{layer}.hook_resid_post.hook_sae_recons",
                f"blocks.{layer}.hook_resid_post.hook_sae_acts_post",
            ],
        )[1]
        #Residual stream
        act_input = extract_last_token_acts(
            batch_out[f"blocks.{layer}.hook_resid_post.hook_sae_input"],
            batch_mask,
            offset,
        )
        #SAE reconstruction
        act_recons = extract_last_token_acts(
            batch_out[f"blocks.{layer}.hook_resid_post.hook_sae_recons"],
            batch_mask,
            offset,
        )
        #SAE error
        act_diff = act_input - act_recons
        #SAE hidden activations post nonlinearity
        act_acts_post = extract_last_token_acts(
            batch_out[f"blocks.{layer}.hook_resid_post.hook_sae_acts_post"],
            batch_mask,
            offset,
        )
        all_feats_input.append(act_input.detach().cpu())
        all_feats_recons.append(act_recons.detach().cpu())
        all_feats_diff.append(act_diff.detach().cpu())
        all_feats_acts_post.append(act_acts_post.detach().cpu())

    feats_input = t.cat(all_feats_input, dim=0)
    feats_recons = t.cat(all_feats_recons, dim=0)
    feats_diff = t.cat(all_feats_diff, dim=0)
    feats_acts_post = t.cat(all_feats_acts_post, dim=0)
    return feats_input, feats_recons, feats_diff, feats_acts_post


def load_model_and_sae(
    model_name: str = "gemma-2-2b",
    sae_release: str = "gemma-scope-2b-pt-res-canonical",
    sae_id: str = "layer_19/width_16k/canonical",
    device: str = "cuda",
) -> tuple[HookedSAETransformer, SAE]:
    """
    Load a transformer model and corresponding SAE
    
    Args:
        model_name: Name of the model to load (e.g. "gemma-2-2b")
        sae_release: SAE release name
        sae_id: SAE ID for the specific layer/configuration
        device: Device to load the model on
        
    Returns:
        Tuple of (model, sae)
    """
    # Load SAE
    sae, _, _ = SAE.from_pretrained(
        release=sae_release, sae_id=sae_id, device=device
    )
    
    # Load model
    model = HookedSAETransformer.from_pretrained(
        model_name, device=device, dtype=t.bfloat16
    )
    
    return model, sae


def prepare_features_for_probing(
    df: pd.DataFrame,
    model: HookedSAETransformer, 
    sae: SAE,
    text_column: str = "prompt",
    layer: int = 19,
    batch_size: int = 8,
    device: str = "cuda",
    offset: int = 1,
) -> dict[str, t.Tensor]:
    """
    Prepare features for probing by tokenizing text and extracting model activations
    
    Args:
        df: DataFrame containing the text data
        model: HookedSAETransformer model
        sae: SAE model 
        text_column: Column in DataFrame containing text to analyze
        layer: Layer to extract activations from
        batch_size: Batch size for processing
        device: Device to use for computation
        offset: Token position offset from the end
        
    Returns:
        Dictionary with keys 'sae_input', 'sae_recons', 'sae_diff', and 'sae_acts_post'
        containing the corresponding activation tensors
    """
    # Tokenize the data
    tokenized = tokenize_data(df, model.tokenizer, text_column=text_column)
    
    # Extract features
    feats_input, feats_recons, feats_diff, feats_acts_post = generate_probing_features(
        tokenized, model, sae, layer=layer, batch_size=batch_size, 
        device=device, offset=offset
    )
    
    # Return features as a dictionary for easier access
    features_map = {
        "sae_input": feats_input,
        "sae_recons": feats_recons,
        "sae_diff": feats_diff, 
        "sae_acts_post": feats_acts_post
    }
    
    return features_map


###############################################################################
# Example Usage
###############################################################################
if __name__ == "__main__":
    # Example of how to use the functions above to prepare data for probing experiments
    from pathlib import Path
    file_path = Path(__file__).parent
    # Set the device
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    device_str = str(device)
    
    # Load model and SAE
    model, sae = load_model_and_sae(
        model_name="gemma-2-2b",
        sae_release="gemma-scope-2b-pt-res-canonical", 
        sae_id="layer_19/width_16k/canonical",
        device=device_str
    )
    
    # Load data
    # Replace 'path/to/data.csv' with the actual path to your dataset
    df = pd.read_csv(file_path/'data'/'149_twt_emotion_happiness.csv')
    
    # Create train-test split
    train_df, test_df = train_test_split_df(df, test_size=0.2, seed=42)
    
    # Prepare features for both train and test sets
    train_features = prepare_features_for_probing(
        train_df, 
        model=model,
        sae=sae,
        text_column="prompt",  # Replace with your text column name
        layer=19,
        batch_size=8,
        device=device_str
    )
    
    test_features = prepare_features_for_probing(
        test_df,
        model=model,
        sae=sae,
        text_column="prompt",  # Replace with your text column name
        layer=19,
        batch_size=8,
        device=device_str
    )
    
    # Now train_features and test_features dictionaries contain:
    # - 'sae_input': Residual stream activations (direct model activations)
    # - 'sae_recons': SAE reconstruction of those activations 
    # - 'sae_diff': Reconstruction error (difference between input and reconstruction)
    # - 'sae_acts_post': SAE hidden layer activations (sparse features)
    
    # You can now design and train various probing models using these features
    # Example: access residual stream activations for training
    train_residual_stream = train_features['sae_input']     # shape: [batch, d_model]
    train_sae_hidden = train_features['sae_acts_post']      # shape: [batch, d_sae_hidden]
    train_recon_error = train_features['sae_diff']          # shape: [batch, d_model]
    
    # Example accessing labels (assuming 'target' column exists in dataframe)
    train_labels = t.tensor(train_df['target'].values)
    test_labels = t.tensor(test_df['target'].values)
    
    # From here, you can implement your own probe training logic
    # using the extracted features and labels
# %%
