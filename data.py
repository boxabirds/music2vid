from sqlalchemy import create_engine, Column, Integer, String, Date, ForeignKey, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy import Enum

Base = declarative_base()


class CompositionState(Enum):
    pending = "pending"
    stemming = "stemming"
    transcription = "transcription"
    pulse_detection = "pulse detection"
    initial_animation_description_generation = "initial animation description generation"
    initial_style_generation = "initial style generation"

class SessionState(Enum):
    pending = "pending"
    lyric_frame_generation = "lyric frame generation"
    key_frame_generation = "key frame generation"
    interpolation_frame_generation = "interpolation frame generation"
    video_generation = "video generation"

class Quality(Enum):
    tiny = "tiny"
    base = "base"
    small = "small"
    medium = "medium"
    large = "large"


class Composition(Base):
    __tablename__ = 'compositions'

    id = Column(Integer, primary_key=True)
    name = Column(String)
    mp3_url = Column(String)
    state = Column(Enum(CompositionState))
    mp3_vocal_stem_url = Column(String)
    wav_drums_url = Column(String)
    transcription_id = Column(Integer, ForeignKey('transcriptions.id'))

    transcription = relationship("Transcription", back_populates="composition")


class Transcription(Base):
    __tablename__ = 'transcriptions'

    id = Column(Integer, primary_key=True)
    quality = Column(Enum(Quality))

    composition = relationship("Composition", back_populates="transcription")
    lines = relationship("TranscriptionLine", back_populates="transcription")


class TranscriptionLine(Base):
    __tablename__ = 'transcription_lines'

    id = Column(Integer, primary_key=True)
    text = Column(String)
    transcription_id = Column(Integer, ForeignKey('transcriptions.id'))

    transcription = relationship("Transcription", back_populates="lines")


class Session(Base):
    __tablename__ = 'sessions'

    id = Column(Integer, primary_key=True)
    name = Column(String)
    creation_date = Column(Date)
    state = Column(Enum(SessionState))
    composition_id = Column(Integer, ForeignKey('compositions.id'))

    composition = relationship("Composition", back_populates="sessions")
    frames = relationship("Frame", back_populates="session")
    markers = relationship("Marker", back_populates="session")


class Frame(Base):
    __tablename__ = 'frames'

    id = Column(Integer, primary_key=True)
    frame_url = Column(String)
    frame_seed = Column(Integer)
    session_id = Column(Integer, ForeignKey('sessions.id'))

    session = relationship("Session", back_populates="frames")


class Marker(Base):
    __tablename__ = 'markers'

    id = Column(Integer, primary_key=True)
    text = Column(String)
    timestamp = Column(String)
    session_id = Column(Integer, ForeignKey('sessions.id'))
    frame_id = Column(Integer, ForeignKey('frames.id'))

    session = relationship("Session", back_populates="markers")
    frame = relationship("Frame", back_populates="markers")

# Establish relationships between entities
Composition.sessions = relationship("Session", back_populates="composition")
Frame.markers = relationship("Marker", back_populates="frame")
