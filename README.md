Format of database.json:
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


Format of image_caption.json
{
  "image_id1": {
    "article_id": ...,
    "title": ...,
    "category": ...,
    "original_caption": ...,
    "generated_caption": ...
  },
  "image_id2": {
    ...
  }
}


Format of image_questions.json
{
  "image_id1": [
    "question1",
    "question2",
    ...
  ],
  "image_id2": [
    "question3",
    "questions4"
  ],
  ...
}


Format of image_vqa.json

{
  "image_id": [
    [
        "question1",
        "answer1"
    ],
    [
        "question2",
        "answer2"
    ],
    ...
  ]
}