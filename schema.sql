CREATE TABLE images (
    id UUID PRIMARY KEY,
    filename VARCHAR(100) NOT NULL,
    embedding VECTOR(2048) NOT NULL,
    total_tokens INT,
    image_tokens INT,
    text_tokens INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
