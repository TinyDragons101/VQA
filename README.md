The format of database.json:

{
  "article_id_using_16hex": // f8097c7d27a8aac6 {
    "url": "...",
    "date": "...", // 2021-07-15T02:46:59Z
    "title": "...",
    "images": [
      {
        "image_id": "16hex",
        "caption": "...",
        "author": "..."
      }
    ],
    "content": "..."
  }, 
  "article_id_using_16hex2": {
    ...
  }
}


The format of image_caption.json

{
    "image_id": {
        "article_id": ...,
        "title": ...,
        "url": ...,
        "original_caption": ...,
        "generated_caption": ...
    }
}


The format of image_vqa.json

{
    "image_id": [
        {
            "question": ...,
            "answer": ...,
        },
        {
            "question": ...,
            "answer": ...,
        },
        {
            ...
        }
    ]
}