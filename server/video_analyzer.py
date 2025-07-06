import os
import json
import logging
import base64
import google.generativeai as genai
from datetime import datetime
from pydantic import BaseModel, Field
from typing import List, Optional

logger = logging.getLogger(__name__)

# Pydantic models for structured output
class ObjectAffected(BaseModel):
    type: str = Field(description="Type of object (e.g., person, fly swatter, fly)")
    start_location_description: str = Field(description="Description of where the object started")
    end_location_description: str = Field(description="Description of where the object ended up")

class ObjectState(BaseModel):
    type: str = Field(description="Type of object (e.g., three drinking glasses, fly swatter, banana bunch)")
    quantity: int = Field(description="Quantity of the object")
    start_location_description: str = Field(description="Description of where the object started")
    start_state: str = Field(description="State of the object at the start (e.g., at rest, two stacked together, five bananas left)")

class ObjectEndState(BaseModel):
    type: str = Field(description="Type of object (e.g., three glasses, fly swatter, banana bunch)")
    end_location_description: str = Field(description="Description of where the object ended up")
    end_state: str = Field(description="State of the object at the end (e.g., in motion, being cleaned, four bananas left)")

class Event(BaseModel):
    description: str = Field(description="Detailed description of what happened in this event")
    objects_affected: List[ObjectAffected] = Field(description="List of objects involved in this event")

class VideoAnalysis(BaseModel):
    object_start_state: List[ObjectState] = Field(description="List of objects and their initial states at the start of the video")
    object_end_state: List[ObjectEndState] = Field(description="List of objects and their final states at the end of the video")
    events: List[Event] = Field(description="List of events that occurred in the video")

class VideoAnalyzer:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        else:
            logger.warning("No Gemini API key provided. Video analysis will be disabled.")
            self.model = None
    
    def analyze_video(self, video_path, metadata_path=None):
        """Analyze a video using Gemini 2.5 Flash with structured output"""
        if not self.model:
            return {"error": "Gemini API key not configured"}
        
        try:
            # Read the video file
            with open(video_path, 'rb') as video_file:
                video_data = video_file.read()
            
            # Create the prompt
            prompt = """
            You are an AI video analysis assistant that receives videos from a program and gives back a structured response about the contents. 
            The goal is to give detailed tracking of all objects that were manipulated. The received response will be used to compile an event log which later on 
            can be used to retrieve the last location of a certain object, the amount of times certain actions were performed, the amount of stock left for a certain product, etc.

            It is important that tracking happens for all objects relevant for tracking, so no stationary objects, and state that is relevant for the specific object. It is better to track too much than too little.
            It is also important to track any events executed by persons, including the amount of times a certain action was performed. 

            You will be given a video as data.

            Here's an example of the expected response format:

            {
              "object_start_state": [
                {
                  "type": "three drinking glasses",
                  "quantity": 3,
                  "start_location_description": "On the counter above the dishwasher",
                  "start_state": "two stacked together, one next to it"
                },
                {
                  "type": "fly swatter",
                  "quantity": 1,
                  "start_location_description": "on the kitchen counter",
                  "start_state": "at rest"
                },
                {
                  "type": "banana bunch",
                  "quantity": 1,
                  "start_location_description": "in the cupboard",
                  "start_state": "five bananas left"
                },
                {
                  "type": "frying pan",
                  "quantity": 1,
                  "start_location_description": "in the sink",
                  "start_state": "dirty"
                }
              ],
              "object_end_state": [
                {
                  "type": "three glasses",
                  "end_location_description": "inside the dishwasher",
                  "end_state": "being cleaned by the dishwasher"
                },
                {
                  "type": "fly swatter",
                  "end_location_description": "carried away by person",
                  "end_state": "in motion"
                },
                {
                  "type": "banana bunch",
                  "end_location_description": "moved out of the view",
                  "end_state": "four bananas left, moving out of the view"
                },
                {
                  "type": "frying pan",
                  "end_location_description": "in the corner kitchen cabinet left below the sink",
                  "end_state": "clean"
                }
              ],
              "events": [
                {
                  "description": "Person walked into the kitchen and grabbed a glass",
                  "objects_affected": [
                    {
                      "type": "person",
                      "start_location_description": "entered view from the right",
                      "end_location_description": "standing in front of the kitchen counter"
                    },
                    {
                      "type": "glass",
                      "start_location_description": "on the kitchen counter",
                      "end_location_description": "carried away by person"
                    }
                  ]
                },
                {
                  "description": "Person filled the glass with water from the kitchen sink",
                  "objects_affected": [
                    {
                      "type": "glass",
                      "start_location_description": "carried by person",
                      "end_location_description": "at the kitchen sink"
                    },
                    {
                      "type": "kitchen sink",
                      "start_location_description": "fixed location",
                      "end_location_description": "fixed location"
                    },
                    {
                      "type": "water",
                      "start_location_description": "inside the kitchen sink",
                      "end_location_description": "inside the glass"
                    }
                  ]
                },
                {
                  "description": "Person drank the water from the glass",
                  "objects_affected": [
                    {
                      "type": "glass",
                      "start_location_description": "at the kitchen sink",
                      "end_location_description": "at the kitchen sink"
                    }
                  ]
                },
                {
                  "description": "Person put the glass on the kitchen counter and walked away",
                  "objects_affected": [
                    {
                      "type": "glass",
                      "start_location_description": "carried by person",
                      "end_location_description": "on the kitchen counter"
                    },
                    {
                      "type": "person",
                      "start_location_description": "standing in front of the kitchen counter",
                      "end_location_description": "moved to the right side of the view"
                    }
                  ]
                }
              ]
            }

            Analyze the video and provide a structured response with the following information:
            - Break down the video into distinct events/actions
            - For each event, describe what happened and what objects were involved
            - Be as detailed as possible in describing object locations, movements, and states
            - Focus on human actions, object interactions, and significant movements

            The descriptions can be much longer if more happened, but try as best as possible to cut up different actions into separate events. Be as extensive as possible in terms of naming all the object and events.
            """
            
            # Create the generation config with structured output
            generation_config = genai.types.GenerationConfig(
                temperature=0.1,
                top_p=0.8,
                top_k=40,
                max_output_tokens=8192,
                response_mime_type="application/json",
                response_schema={
                    "type": "object",
                    "properties": {
                        "object_start_state": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "type": {"type": "string"},
                                    "quantity": {"type": "integer"},
                                    "start_location_description": {"type": "string"},
                                    "start_state": {"type": "string"}
                                },
                                "required": ["type", "quantity", "start_location_description", "start_state"]
                            }
                        },
                        "object_end_state": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "type": {"type": "string"},
                                    "end_location_description": {"type": "string"},
                                    "end_state": {"type": "string"}
                                },
                                "required": ["type", "end_location_description", "end_state"]
                            }
                        },
                        "events": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "description": {"type": "string"},
                                    "objects_affected": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "type": {"type": "string"},
                                                "start_location_description": {"type": "string"},
                                                "end_location_description": {"type": "string"}
                                            },
                                            "required": ["type", "start_location_description", "end_location_description"]
                                        }
                                    }
                                },
                                "required": ["description", "objects_affected"]
                            }
                        }
                    },
                    "required": ["object_start_state", "object_end_state", "events"]
                }
            )
            
            # Analyze the video
            logger.info(f"Starting video analysis for: {video_path}")
            response = self.model.generate_content(
                [prompt, {"mime_type": "video/mp4", "data": video_data}],
                generation_config=generation_config
            )
            
            # Parse the structured response
            try:
                # Extract JSON from the response text (most reliable method)
                response_text = response.text
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                
                if start_idx != -1 and end_idx != 0:
                    json_str = response_text[start_idx:end_idx]
                    analysis_dict = json.loads(json_str)
                    
                    # Add metadata
                    analysis_dict["analysis_metadata"] = {
                        "video_path": video_path,
                        "analysis_timestamp": datetime.now().isoformat(),
                        "model_used": "gemini-2.0-flash-exp",
                        "structured_output": True
                    }
                    
                    # Save the analysis to a JSON file
                    analysis_path = video_path.replace('.mp4', '_analysis.json')
                    with open(analysis_path, 'w', encoding='utf-8') as f:
                        json.dump(analysis_dict, f, indent=2, ensure_ascii=False)
                    
                    logger.info(f"Video analysis completed and saved to: {analysis_path}")
                    return analysis_dict
                else:
                    raise ValueError("No JSON content found in response")
                
            except Exception as e:
                logger.error(f"Error parsing structured response: {e}")
                # Fallback to raw response if structured parsing fails
                analysis_data = {
                    "raw_response": response.text,
                    "parse_error": str(e),
                    "analysis_metadata": {
                        "video_path": video_path,
                        "analysis_timestamp": datetime.now().isoformat(),
                        "model_used": "gemini-2.0-flash-exp",
                        "structured_output": False
                    }
                }
                
                # Save fallback analysis
                analysis_path = video_path.replace('.mp4', '_analysis.json')
                with open(analysis_path, 'w', encoding='utf-8') as f:
                    json.dump(analysis_data, f, indent=2, ensure_ascii=False)
                
                return analysis_data
            
        except Exception as e:
            logger.error(f"Error analyzing video {video_path}: {e}")
            return {"error": str(e)}
    
    def get_analysis(self, video_path):
        """Get existing analysis for a video"""
        analysis_path = video_path.replace('.mp4', '_analysis.json')
        if os.path.exists(analysis_path):
            try:
                with open(analysis_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error reading analysis file {analysis_path}: {e}")
                return {"error": str(e)}
        else:
            return {"error": "Analysis not found"}

    def ask_question_about_video(self, video_path, question):
        """Ask a specific question about a video using Gemini"""
        if not self.model:
            return {"error": "Gemini API key not configured"}
        
        try:
            # Read the video file
            with open(video_path, 'rb') as video_file:
                video_data = video_file.read()
            
            # Create the prompt with the specific question
            prompt = f"""
            You are analyzing a video to answer a specific question. Please watch the video carefully and provide a detailed answer.

            Question: {question}

            Please provide a comprehensive answer based on what you observe in the video. Be as specific as possible about:
            - What you see happening
            - When things occur (if timing is relevant)
            - Where objects or people are located
            - Any details that are relevant to the question

            Answer the question as accurately as possible based on the video content.
            """
            
            # Create the generation config
            generation_config = genai.types.GenerationConfig(
                temperature=0.1,
                top_p=0.8,
                top_k=40,
                max_output_tokens=4096,
            )
            
            # Ask the question
            logger.info(f"Asking question about video: {video_path}")
            logger.info(f"Question: {question}")
            
            response = self.model.generate_content(
                [prompt, {"mime_type": "video/mp4", "data": video_data}],
                generation_config=generation_config
            )
            
            # Get the response
            answer = response.text
            
            # Create response data
            question_data = {
                "question": question,
                "answer": answer,
                "video_path": video_path,
                "timestamp": datetime.now().isoformat(),
                "model_used": "gemini-2.0-flash-exp"
            }
            
            logger.info(f"Question answered successfully for: {video_path}")
            return question_data
            
        except Exception as e:
            logger.error(f"Error asking question about video {video_path}: {e}")
            return {"error": str(e)}

# Global instance
video_analyzer = None

def get_video_analyzer():
    global video_analyzer
    if video_analyzer is None:
        video_analyzer = VideoAnalyzer()
    return video_analyzer 