CREATE TABLE faces (
    imdb_id        INT,
    frame_num      INT,
    face_num       INT,
    x1             INT,
    x2             INT,
    y1             INT,
    y2             INT,
    img_height     INT,
    img_width      INT,
    area           INT,
    pct_of_frame   FLOAT,
    season         INT,
    episode        INT,
    encoding       VARCHAR(255),
    character_name VARCHAR(255),
    cast_id        INT
)
