import os
import base64
import uuid
import numpy as np
import psycopg2
from datetime import datetime
from byteplussdkarkruntime import Ark
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

# Database configuration
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "postgres"
DB_USER = "postgres"


def get_db_connection():
    """Create a connection to PostgreSQL database"""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
        )
        return conn
    except Exception as e:
        print(f"Failed to connect to database: {str(e)}")
        raise


def save_embedding_to_db(
    filename, embedding, total_tokens=None, image_tokens=None, text_tokens=None
):
    """Save image embedding to PostgreSQL database"""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        image_id = str(uuid.uuid4())
        embedding_list = embedding.tolist()

        cursor.execute(
            """
            INSERT INTO images (id, filename, embedding, total_tokens, image_tokens, text_tokens, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                image_id,
                filename,
                embedding_list,
                total_tokens,
                image_tokens,
                text_tokens,
                datetime.now(),
            ),
        )
        conn.commit()
        return image_id
    except Exception as e:
        conn.rollback()
        print(f" - Failed to save to DB: {filename} - {str(e)}")
        raise
    finally:
        cursor.close()
        conn.close()


def get_embedding(input_data, input_type="image_base64"):
    """Call the BytePlus API to get the vector representation of an image"""
    client = Ark(api_key=os.environ.get("ARK_API_KEY"))
    if input_type == "image_url":
        input_item = {"type": "image_url", "image_url": {"url": input_data}}
    elif input_type == "image_base64":
        # For base64 encoded images, use image_url type with data URI format
        input_item = {"type": "image_url", "image_url": {"url": input_data}}
    else:
        raise ValueError(
            "Only 'image_url' or 'image_base64' are supported for input type"
        )

    try:
        resp = client.multimodal_embeddings.create(
            model="skylark-embedding-vision-250615",
            encoding_format="float",
            input=[input_item],
        )
        # If resp is an object, convert to dict
        if hasattr(resp, "__dict__"):
            resp = resp.__dict__
        # If resp.data is an object, convert to dict
        if hasattr(resp.get("data", {}), "__dict__"):
            resp["data"] = resp["data"].__dict__
        if (
            "data" in resp
            and isinstance(resp["data"], dict)
            and "embedding" in resp["data"]
        ):
            embedding = resp["data"]["embedding"]
            embedding = np.array(embedding).flatten()

            usage_info = {
                "total_tokens": None,
                "image_tokens": None,
                "text_tokens": None,
            }
            if "usage" in resp and isinstance(resp["usage"], dict):
                usage = resp["usage"]
                usage_info["total_tokens"] = usage.get(
                    "total_tokens", usage.get("prompt_tokens", None)
                )
                prompt_tokens_details = usage.get("prompt_tokens_details", {})
                usage_info["image_tokens"] = prompt_tokens_details.get(
                    "image_tokens", None
                )
                usage_info["text_tokens"] = prompt_tokens_details.get(
                    "text_tokens", None
                )

            return embedding, usage_info
        else:
            raise ValueError(
                "API response format is not as expected, cannot obtain embedding vector"
            )
    except Exception as e:
        print(f"Failed to get vector, input type: {input_type}, error: {str(e)}")
        raise


def load_images_from_directory(directory_path):
    """Load all image files from a directory"""
    image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
    image_paths = []

    if not os.path.exists(directory_path):
        raise ValueError(f"Directory not found: {directory_path}")

    for filename in os.listdir(directory_path):
        if os.path.splitext(filename)[1].lower() in image_extensions:
            image_paths.append(os.path.join(directory_path, filename))

    if not image_paths:
        raise ValueError(f"No images found in {directory_path}")

    return sorted(image_paths)


def encode_image(image_path):
    """Base64 encode an image file with proper data URI format"""
    # Determine image format from file extension
    file_ext = os.path.splitext(image_path)[1].lower().lstrip(".")
    # Map common extensions to MIME types
    format_map = {
        "jpg": "jpeg",
        "jpeg": "jpeg",
        "png": "png",
        "gif": "gif",
        "bmp": "bmp",
        "webp": "webp",
    }
    image_format = format_map.get(file_ext, file_ext)

    with open(image_path, "rb") as image_file:
        base64_str = base64.b64encode(image_file.read()).decode("utf-8")

    # Return in the format: data:image/<FORMAT>;base64,<BASE64_ENCODING>
    return f"data:image/{image_format};base64,{base64_str}"


def generate_image_embeddings(image_paths, save_to_db=True):
    """Batch generate image embeddings from local files and optionally save to database"""
    print(f"[1/2] Start generating embeddings for {len(image_paths)} images...")
    embeddings = []
    for i, image_path in enumerate(image_paths):
        try:
            # Encode image to base64
            base64_image = encode_image(image_path)
            # Get embedding from API
            embedding, usage_info = get_embedding(base64_image, "image_base64")
            filename = os.path.basename(image_path)

            # Save to database if requested
            if save_to_db:
                save_embedding_to_db(
                    filename,
                    embedding,
                    total_tokens=usage_info["total_tokens"],
                    image_tokens=usage_info["image_tokens"],
                    text_tokens=usage_info["text_tokens"],
                )

            embeddings.append(
                {"image_path": image_path, "filename": filename, "embedding": embedding}
            )
            print(f" [{i+1}/{len(image_paths)}] Success: {filename}")
        except Exception as e:
            print(
                f" [{i+1}/{len(image_paths)}] Failed: {os.path.basename(image_path)} - {str(e)}"
            )
            continue

    if not embeddings:
        raise ValueError("All image embeddings generation failed")
    print(f"[2/2] Completed: {len(embeddings)} valid embeddings")
    return embeddings


def search_similar_images(query_embedding, embeddings, top_n=1, query_type="text"):
    """Search for images most similar to the query vector"""
    print(f"\n[3/3] Start searching for images most similar to the {query_type}...")
    results = []
    # Convert the query vector into a NumPy array with a correct number of dimensions.
    query_vec = np.array(query_embedding).reshape(1, -1)
    for item in embeddings:
        # Convert the vector into a two-dimensional NumPy array for similarity calculation.
        item_vec = np.array(item["embedding"]).reshape(1, -1)

        similarity = cosine_similarity(query_vec, item_vec)[0][0]
        results.append(
            {
                "filename": item.get("filename", item.get("image_url")),
                "image_path": item.get("image_path"),
                "similarity": similarity,
            }
        )

    results.sort(key=lambda x: x["similarity"], reverse=True)
    print(f" - Similarity calculation completed, total {len(results)} results")
    return results[:top_n]


if __name__ == "__main__":
    image_dir = "data/cat"

    try:
        # Load all images from the directory
        image_paths = load_images_from_directory(image_dir)
        print(f"Found {len(image_paths)} images in {image_dir}\n")

        # Convert images to vectors and save to database
        image_embs = generate_image_embeddings(image_paths, save_to_db=True)

    except Exception as e:
        print(f"Program failed: {e}")
