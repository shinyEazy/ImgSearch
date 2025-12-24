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


def generate_image_embeddings(image_path):
    """Batch generate image embeddings from local files and optionally save to database"""
    print(f"[1/2] Start generating embeddings for 1 image...")
    embeddings = []
    try:
        # Encode image to base64
        base64_image = encode_image(image_path)
        # Get embedding from API
        embedding, usage_info = get_embedding(base64_image, "image_base64")
        filename = os.path.basename(image_path)

        embeddings.append(
            {"image_path": image_path, "filename": filename, "embedding": embedding}
        )
    except Exception as e:
        print(f" [1/1] Failed: {os.path.basename(image_path)} - {str(e)}")
    if not embeddings:
        raise ValueError("All image embeddings generation failed")
    print(f"[2/2] Completed: {len(embeddings)} valid embeddings - usage: {usage_info}")
    return embeddings


def get_topk_similar_from_db(query_embedding, k=5):
    """Query Postgres (with pgvector) for top k most similar images by embedding."""
    conn = get_db_connection()
    cur = conn.cursor()
    # pgvector: embedding <-> %s for cosine distance, order by ascending (most similar first)
    sql = f"""
        SELECT
            filename,
            1 - (embedding <=> %s) / 2 AS similarity
        FROM images
        ORDER BY embedding <=> %s
        LIMIT {k};
    """
    # Ensure query_embedding is a list/array of floats
    if isinstance(query_embedding, np.ndarray):
        query_embedding = query_embedding.astype(float).tolist()
    else:
        query_embedding = [float(x) for x in query_embedding]
    # Convert to pgvector string format: '[0.1,0.2,0.3]'
    vector_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
    cur.execute(sql, (vector_str, vector_str))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    # Return list of dicts: [{filename, similarity}]
    return [{"filename": row[0], "similarity": row[1]} for row in rows]


def find_topk_similar_images(image_path, k=5):
    """Given an image file, return top k most similar image filenames from DB using pgvector."""
    embedding_list = generate_image_embeddings(image_path)
    query_embedding = embedding_list[0]["embedding"]
    topk_results = get_topk_similar_from_db(query_embedding, k)
    return topk_results


if __name__ == "__main__":
    animal = input("Enter an animal name to use its image: ").strip().lower()
    image_path = f"data/test/{animal}_t.jpg"
    k = 10

    try:
        topk = find_topk_similar_images(image_path, k)
        print(f"Top {k} similar images:")
        for i, item in enumerate(topk, start=1):
            print(f"{i}. {item['filename']} - {item['similarity']:.2f}")
    except Exception as e:
        print(f"Program failed: {e}")
