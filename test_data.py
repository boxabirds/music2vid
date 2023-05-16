from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from data import Base, Composition, CompositionState, Quality, Session, SessionState, Transcription, TranscriptionLine

# Use the models you already defined (Quality, Composition, etc.)

# Create an SQLite database file and establish a connection
engine = create_engine('sqlite:///compositions.db', echo=True)

# Create the tables in the database
Base.metadata.create_all(engine)

# Create a session using the sessionmaker class
Session = sessionmaker(bind=engine)
session = Session()

# Add a new Composition object to the session
new_composition = Composition(
    name="My Composition",
    mp3_url="https://example.com/my_composition.mp3",
    state=CompositionState.pending,
    mp3_vocal_stem_url="https://example.com/my_composition_vocal.mp3",
    wav_drums_url="https://example.com/my_composition_drums.wav"
)

session.add(new_composition)

# Create a new Session object with the appropriate state value
new_session = Session(
    state=SessionState.pending
)

# Create a new Transcription object with the required information
new_transcription = Transcription(
    content="Transcription content here"
)

# Associate the new Session and Transcription objects with the Composition object
new_composition.session = new_session
new_composition.transcription = new_transcription

# Add the new Session and Transcription objects to the session
session.add(new_session)
session.add(new_transcription)

# Commit the changes to the database
session.commit()

# Close the session
session.close()
