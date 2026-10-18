"""
Image Generator

Generate poster/slides images from ContentPlan.
"""
import os
import json
import base64
import time
import logging
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import requests
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import GenerationInput
from .content_planner import ContentPlan, Section
from ..prompts.image_generation import (
    STYLE_PROCESS_PROMPT,
    FORMAT_POSTER,
    FORMAT_SLIDE,
    POSTER_STYLE_HINTS,
    SLIDE_STYLE_HINTS,
    SLIDE_LAYOUTS_ACADEMIC,
    SLIDE_LAYOUTS_DORAEMON,
    SLIDE_LAYOUTS_DEFAULT,
    SLIDE_COMMON_STYLE_RULES,
    POSTER_COMMON_STYLE_RULES,
    VISUALIZATION_HINTS,
    CONSISTENCY_HINT,
    SLIDE_FIGURE_HINT,
    POSTER_FIGURE_HINT,
)
from .academic_compositor import render_academic_poster, render_academic_slide


@dataclass
class GeneratedImage:
    """Generated image result."""
    section_id: str
    image_data: bytes
    mime_type: str


@dataclass
class ProcessedStyle:
    """Processed custom style from LLM."""
    style_name: str       # e.g., "Cyberpunk sci-fi style with high-tech aesthetic"
    color_tone: str       # e.g., "dark background with neon accents"
    special_elements: str # e.g., "Characters appear as guides" or ""
    decorations: str      # e.g., "subtle grid pattern" or ""
    valid: bool
    error: Optional[str] = None


def process_custom_style(client: OpenAI, user_style: str, model: str = None) -> ProcessedStyle:
    """Process user's custom style request with LLM."""
    model = model or os.getenv("LLM_MODEL", "openai/gpt-4o-mini")
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": STYLE_PROCESS_PROMPT.format(user_style=user_style)}],
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
        return ProcessedStyle(
            style_name=result.get("style_name", ""),
            color_tone=result.get("color_tone", ""),
            special_elements=result.get("special_elements", ""),
            decorations=result.get("decorations", ""),
            valid=result.get("valid", False),
            error=result.get("error"),
        )
    except Exception as e:
        return ProcessedStyle(style_name="", color_tone="", special_elements="", decorations="", valid=False, error=str(e))


class ImageGenerator:
    """Generate poster/slides images from ContentPlan."""
    
    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        response_mime_type: Optional[str] = None,
        google_api_base_url: Optional[str] = None,
    ):
        self.provider = (provider or os.getenv("IMAGE_GEN_PROVIDER", "openrouter")).lower()
        self.api_key = api_key or os.getenv("IMAGE_GEN_API_KEY", "")
        self.base_url = base_url or os.getenv("IMAGE_GEN_BASE_URL", "https://openrouter.ai/api/v1")
        self.google_api_base_url = (google_api_base_url or os.getenv("GOOGLE_GENAI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")).rstrip("/")
        self.response_mime_type = response_mime_type or os.getenv("IMAGE_GEN_RESPONSE_MIME_TYPE", "text/plain")
        self.model = model or os.getenv("IMAGE_GEN_MODEL")
        self.comfyui_base_url = os.getenv("IMAGE_GEN_COMFYUI_BASE_URL", "http://127.0.0.1:8188").rstrip("/")
        self.comfyui_workflow_path = os.getenv("IMAGE_GEN_COMFYUI_WORKFLOW", "")
        self.comfyui_prompt_node = os.getenv("IMAGE_GEN_COMFYUI_PROMPT_NODE", "")
        self.comfyui_prompt_input = os.getenv("IMAGE_GEN_COMFYUI_PROMPT_INPUT", "text")
        self.comfyui_output_node = os.getenv("IMAGE_GEN_COMFYUI_OUTPUT_NODE", "")
        self.comfyui_timeout_seconds = float(os.getenv("IMAGE_GEN_COMFYUI_TIMEOUT_SECONDS", "300"))
        self.compose_academic = os.getenv("IMAGE_GEN_COMPOSE_ACADEMIC", "1").strip().lower() not in {
            "0", "false", "no", "off"
        }
        
        if not self.model:
            if self.provider == "google":
                # Official Gemini API image-capable default
                self.model = "models/gemini-1.5-flash"
            else:
                self.model = "google/gemini-3-pro-image-preview"
        
        if self.provider == "openrouter":
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        elif self.provider == "google":
            self.client = None
        elif self.provider == "comfyui":
            self.client = None
            if not self.comfyui_workflow_path:
                raise ValueError(
                    "IMAGE_GEN_PROVIDER=comfyui requires IMAGE_GEN_COMFYUI_WORKFLOW "
                    "pointing to a ComfyUI API-format workflow JSON file."
                )
            workflow_path = Path(self.comfyui_workflow_path).expanduser()
            if not workflow_path.is_file():
                raise ValueError(f"ComfyUI workflow file does not exist: {workflow_path}")
            if not self.comfyui_prompt_node:
                raise ValueError("Set IMAGE_GEN_COMFYUI_PROMPT_NODE to the workflow node containing the positive prompt.")
            if not self.comfyui_base_url.startswith(("http://", "https://")):
                raise ValueError("IMAGE_GEN_COMFYUI_BASE_URL must be an http(s) URL.")
        else:
            raise ValueError(f"Unsupported image generation provider: {self.provider}")
    
    def generate(
        self,
        plan: ContentPlan,
        gen_input: GenerationInput,
        max_workers: int = 1,
        save_callback = None,
    ) -> List[GeneratedImage]:
        """
        Generate images from ContentPlan.
        
        Args:
            plan: ContentPlan from ContentPlanner
            gen_input: GenerationInput with config and origin
            max_workers: Maximum parallel workers for slides (3rd+ slides run in parallel)
            save_callback: Optional callback function(generated_image, index, total) called after each image
        
        Returns:
            List of GeneratedImage (1 for poster, N for slides)
        """
        figure_images = self._load_figure_images(plan, gen_input.origin.base_path)
        style_name = gen_input.config.style.value
        custom_style = gen_input.config.custom_style
        
        # Process custom style with LLM if needed
        processed_style = None
        if style_name == "custom" and custom_style:
            processed_style = process_custom_style(self.client, custom_style)
            if not processed_style.valid:
                raise ValueError(f"Invalid custom style: {processed_style.error}")
        
        all_sections_md = self._format_sections_markdown(plan)
        all_images = self._filter_images(plan.sections, figure_images)
        
        if (
            plan.output_type == "poster"
            and style_name == "academic"
            and self.compose_academic
        ):
            result = self._generate_academic_poster(
                plan,
                figure_images,
                gen_input.config.poster_density.value,
            )
            if save_callback and result:
                save_callback(result[0], 0, 1)
            return result
        if plan.output_type == "poster":
            result = self._generate_poster(style_name, processed_style, all_sections_md, all_images)
            if save_callback and result:
                save_callback(result[0], 0, 1)
            return result
        if self.provider == "comfyui" and style_name == "academic" and self.compose_academic:
            return self._generate_academic_slides(plan, figure_images, save_callback)
        else:
            return self._generate_slides(plan, style_name, processed_style, all_sections_md, figure_images, max_workers, save_callback)

    def _generate_academic_slides(self, plan, figure_images, save_callback=None) -> List[GeneratedImage]:
        """Compose readable academic slides from exact paper evidence.

        Local diffusion checkpoints are kept available for custom or poster
        styles, but they should not rasterize scientific labels and tables.
        """
        results = []
        total = len(plan.sections)
        for index, section in enumerate(plan.sections):
            image_data = render_academic_slide(section, index, total, figure_images)
            generated = GeneratedImage(section_id=section.id, image_data=image_data, mime_type="image/png")
            results.append(generated)
            if save_callback:
                save_callback(generated, index, total)
        return results

    def _generate_academic_poster(self, plan, figure_images, density: str) -> List[GeneratedImage]:
        """Compose a readable academic poster from exact paper evidence.

        Diffusion is intentionally bypassed here: it is useful for visual
        concepts, but it cannot reliably render the poster's scientific text,
        tables, or numerical labels.  The local compositor keeps those exact.
        """
        image_data = render_academic_poster(plan, figure_images, density=density)
        return [GeneratedImage(section_id="poster", image_data=image_data, mime_type="image/png")]
    
    def _generate_poster(self, style_name, processed_style: Optional[ProcessedStyle], sections_md, images) -> List[GeneratedImage]:
        """Generate 1 poster image."""
        prompt = self._build_poster_prompt(
            format_prefix=FORMAT_POSTER,
            style_name=style_name,
            processed_style=processed_style,
            sections_md=sections_md,
        )
        
        image_data, mime_type = self._call_model(prompt, images)
        return [GeneratedImage(section_id="poster", image_data=image_data, mime_type=mime_type)]
    
    def _generate_slides(self, plan, style_name, processed_style: Optional[ProcessedStyle], all_sections_md, figure_images, max_workers: int, save_callback=None) -> List[GeneratedImage]:
        """Generate N slide images (slides 1-2 sequential, 3+ parallel)."""
        results = []
        total = len(plan.sections)
        
        # Select layout rules based on style
        if style_name == "custom":
            layouts = SLIDE_LAYOUTS_DEFAULT
        elif style_name == "doraemon":
            layouts = SLIDE_LAYOUTS_DORAEMON
        else:
            layouts = SLIDE_LAYOUTS_ACADEMIC
        
        style_ref_image = None  # Store 2nd slide as reference for all subsequent slides
        
        # Generate first 2 slides sequentially (slide 1: no ref, slide 2: becomes ref)
        for i in range(min(2, total)):
            section = plan.sections[i]
            section_md = self._format_single_section_markdown(section, plan)
            layout_rule = layouts.get(section.section_type, layouts["content"])
            
            prompt = self._build_slide_prompt(
                style_name=style_name,
                processed_style=processed_style,
                sections_md=section_md,
                layout_rule=layout_rule,
                slide_info=f"Slide {i+1} of {total}",
                context_md=all_sections_md,
            )
            
            section_images = self._filter_images([section], figure_images)
            reference_images = []
            if style_ref_image:
                reference_images.append(style_ref_image)
            reference_images.extend(section_images)
            
            image_data, mime_type = self._call_model(prompt, reference_images)
            
            # Save 2nd slide (i=1) as style reference
            if i == 1:
                style_ref_image = {
                    "figure_id": "Reference Slide",
                    "caption": "STRICTLY MAINTAIN: same background color, same accent color, same font style, same chart/icon style. Keep visual consistency.",
                    "base64": base64.b64encode(image_data).decode("utf-8"),
                    "mime_type": mime_type,
                }
            
            generated_img = GeneratedImage(section_id=section.id, image_data=image_data, mime_type=mime_type)
            results.append(generated_img)
            
            # Save immediately if callback provided
            if save_callback:
                save_callback(generated_img, i, total)
        
        # Generate remaining slides in parallel (from 3rd onwards)
        if total > 2:
            results_dict = {}
            
            def generate_single(i, section):
                section_md = self._format_single_section_markdown(section, plan)
                layout_rule = layouts.get(section.section_type, layouts["content"])
                
                prompt = self._build_slide_prompt(
                    style_name=style_name,
                    processed_style=processed_style,
                    sections_md=section_md,
                    layout_rule=layout_rule,
                    slide_info=f"Slide {i+1} of {total}",
                    context_md=all_sections_md,
                )
                
                section_images = self._filter_images([section], figure_images)
                reference_images = [style_ref_image] if style_ref_image else []
                reference_images.extend(section_images)
                
                image_data, mime_type = self._call_model(prompt, reference_images)
                return i, GeneratedImage(section_id=section.id, image_data=image_data, mime_type=mime_type)
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(generate_single, i, plan.sections[i]): i
                    for i in range(2, total)
                }
                
                for future in as_completed(futures):
                    idx, generated_img = future.result()
                    results_dict[idx] = generated_img
                    
                    # Save immediately if callback provided
                    if save_callback:
                        save_callback(generated_img, idx, total)
            
            # Append in order
            for i in range(2, total):
                results.append(results_dict[i])
        
        return results
    
    def _format_custom_style_for_poster(self, ps: ProcessedStyle) -> str:
        """Format ProcessedStyle into style hints string for poster."""
        parts = [
            ps.style_name + ".",
            "English text only.",
            "Use ROUNDED sans-serif fonts for ALL text.",
            "Characters should react to or interact with the content, with appropriate poses/actions and sizes - not just decoration."
            f"LIMITED COLOR PALETTE (3-4 colors max): {ps.color_tone}.",
            POSTER_COMMON_STYLE_RULES,
        ]
        if ps.special_elements:
            parts.append(ps.special_elements + ".")
        return " ".join(parts)
    
    def _format_custom_style_for_slide(self, ps: ProcessedStyle) -> str:
        """Format ProcessedStyle into style hints string for slide."""
        parts = [
            ps.style_name + ".",
            "English text only.",
            "Use ROUNDED sans-serif fonts for ALL text.",
            "Characters should react to or interact with the content, with appropriate poses/actions and sizes - not just decoration.",
            f"LIMITED COLOR PALETTE (3-4 colors max): {ps.color_tone}.",
            SLIDE_COMMON_STYLE_RULES,
        ]
        if ps.special_elements:
            parts.append(ps.special_elements + ".")
        return " ".join(parts)
    
    def _build_poster_prompt(self, format_prefix, style_name, processed_style: Optional[ProcessedStyle], sections_md) -> str:
        """Build prompt for poster."""
        parts = [format_prefix]
        
        if style_name == "custom" and processed_style:
            parts.append(f"Style: {self._format_custom_style_for_poster(processed_style)}")
            if processed_style.decorations:
                parts.append(f"Decorations: {processed_style.decorations}")
        else:
            parts.append(POSTER_STYLE_HINTS.get(style_name, POSTER_STYLE_HINTS["academic"]))
        
        parts.append(VISUALIZATION_HINTS)
        parts.append(POSTER_FIGURE_HINT)
        parts.append(f"---\nContent:\n{sections_md}")
        
        return "\n\n".join(parts)
    
    def _build_slide_prompt(self, style_name, processed_style: Optional[ProcessedStyle], sections_md, layout_rule, slide_info, context_md) -> str:
        """Build prompt for slide with layout rules and consistency."""
        parts = [FORMAT_SLIDE]
        
        if style_name == "custom" and processed_style:
            parts.append(f"Style: {self._format_custom_style_for_slide(processed_style)}")
        else:
            parts.append(SLIDE_STYLE_HINTS.get(style_name, SLIDE_STYLE_HINTS["academic"]))
        
        # Add layout rule, then decorations if custom style
        parts.append(layout_rule)
        if style_name == "custom" and processed_style and processed_style.decorations:
            parts.append(f"Decorations: {processed_style.decorations}")
        
        parts.append(VISUALIZATION_HINTS)
        parts.append(CONSISTENCY_HINT)
        parts.append(SLIDE_FIGURE_HINT)
        
        parts.append(slide_info)
        parts.append(f"---\nFull presentation context:\n{context_md}")
        parts.append(f"---\nThis slide content:\n{sections_md}")
        
        return "\n\n".join(parts)
    
    def _format_sections_markdown(self, plan: ContentPlan) -> str:
        """Format all sections as markdown."""
        parts = []
        for section in plan.sections:
            parts.append(self._format_single_section_markdown(section, plan))
        return "\n\n---\n\n".join(parts)
    
    def _format_single_section_markdown(self, section: Section, plan: ContentPlan) -> str:
        """Format a single section as markdown."""
        lines = [f"## {section.title}", "", section.content]
        
        for ref in section.tables:
            table = plan.tables_index.get(ref.table_id)
            if table:
                focus_str = f" (focus: {ref.focus})" if ref.focus else ""
                lines.append("")
                lines.append(f"**{ref.table_id}**{focus_str}:")
                lines.append(ref.extract if ref.extract else table.html_content)
        
        for ref in section.figures:
            fig = plan.figures_index.get(ref.figure_id)
            if fig:
                focus_str = f" (focus: {ref.focus})" if ref.focus else ""
                caption = f": {fig.caption}" if fig.caption else ""
                lines.append("")
                lines.append(f"**{ref.figure_id}**{focus_str}{caption}")
                lines.append("[Image attached]")
        
        return "\n".join(lines)
    
    def _load_figure_images(self, plan: ContentPlan, base_path: str) -> List[dict]:
        """Load figure images as base64."""
        images = []
        mime_map = {
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png", ".webp": "image/webp", ".gif": "image/gif"
        }
        
        for fig_id, fig in plan.figures_index.items():
            img_path = Path(fig.image_path)
            if not img_path.exists() and base_path:
                img_path = Path(base_path) / fig.image_path
            
            if not img_path.exists():
                continue
            
            mime_type = mime_map.get(img_path.suffix.lower(), "image/jpeg")
            
            try:
                with open(img_path, "rb") as f:
                    img_data = base64.b64encode(f.read()).decode("utf-8")
                images.append({
                    "figure_id": fig_id,
                    "caption": fig.caption,
                    "base64": img_data,
                    "mime_type": mime_type,
                })
            except Exception:
                continue
        
        return images
    
    def _filter_images(self, sections: List[Section], figure_images: List[dict]) -> List[dict]:
        """Filter images used in given sections."""
        used_ids = set()
        for section in sections:
            for ref in section.figures:
                used_ids.add(ref.figure_id)
        return [img for img in figure_images if img.get("figure_id") in used_ids]
    
    def _call_model(self, prompt: str, reference_images: List[dict]) -> tuple:
        """Call image generation provider based on configuration."""
        if self.provider == "google":
            return self._call_model_google(prompt, reference_images)
        if self.provider == "comfyui":
            return self._call_model_comfyui(prompt)
        return self._call_model_openrouter(prompt, reference_images)

    def _call_model_comfyui(self, prompt: str) -> tuple:
        """Run a configured ComfyUI API workflow and return its first selected image."""
        try:
            workflow = json.loads(Path(self.comfyui_workflow_path).expanduser().read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not load ComfyUI workflow JSON: {exc}") from exc
        if not isinstance(workflow, dict):
            raise ValueError("ComfyUI workflow must be an API-format JSON object keyed by node ID.")

        prompt_node = workflow.get(str(self.comfyui_prompt_node))
        if not isinstance(prompt_node, dict) or not isinstance(prompt_node.get("inputs"), dict):
            raise ValueError(
                f"ComfyUI prompt node {self.comfyui_prompt_node!r} is missing or has no inputs. "
                "Export the workflow in API format and check IMAGE_GEN_COMFYUI_PROMPT_NODE."
            )
        if self.comfyui_prompt_input not in prompt_node["inputs"]:
            raise ValueError(
                f"Prompt input {self.comfyui_prompt_input!r} is not present on ComfyUI node "
                f"{self.comfyui_prompt_node!r}. Set IMAGE_GEN_COMFYUI_PROMPT_INPUT to the correct input name."
            )
        prompt_node["inputs"][self.comfyui_prompt_input] = prompt

        client_id = secrets.token_hex(16)
        try:
            queued = requests.post(
                f"{self.comfyui_base_url}/prompt",
                json={"prompt": workflow, "client_id": client_id},
                timeout=30,
            )
            queued.raise_for_status()
            prompt_id = queued.json().get("prompt_id")
            if not prompt_id:
                raise RuntimeError("ComfyUI accepted the workflow without returning a prompt_id.")
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Could not queue the local ComfyUI workflow at {self.comfyui_base_url}. "
                "Confirm ComfyUI is running and IMAGE_GEN_COMFYUI_BASE_URL is correct."
            ) from exc

        deadline = time.monotonic() + self.comfyui_timeout_seconds
        while time.monotonic() < deadline:
            try:
                history_response = requests.get(
                    f"{self.comfyui_base_url}/history/{prompt_id}", timeout=30
                )
                history_response.raise_for_status()
                history = history_response.json().get(prompt_id, {})
            except requests.RequestException as exc:
                raise RuntimeError("Could not read the ComfyUI generation result.") from exc

            outputs = history.get("outputs", {})
            if outputs:
                output_nodes = [str(self.comfyui_output_node)] if self.comfyui_output_node else list(outputs)
                image_info = None
                for node_id in output_nodes:
                    node_output = outputs.get(node_id, {})
                    images = node_output.get("images", [])
                    if images:
                        image_info = images[0]
                        break
                if image_info is None:
                    raise RuntimeError(
                        "ComfyUI workflow completed but no image was found. Add a SaveImage output node "
                        "and optionally set IMAGE_GEN_COMFYUI_OUTPUT_NODE to its node ID."
                    )
                try:
                    image_response = requests.get(
                        f"{self.comfyui_base_url}/view",
                        params={
                            "filename": image_info["filename"],
                            "subfolder": image_info.get("subfolder", ""),
                            "type": image_info.get("type", "output"),
                        },
                        timeout=60,
                    )
                    image_response.raise_for_status()
                except (KeyError, requests.RequestException) as exc:
                    raise RuntimeError("ComfyUI returned image metadata, but the image could not be downloaded.") from exc
                mime_type = image_response.headers.get("Content-Type", "image/png").split(";", 1)[0]
                if not mime_type.startswith("image/"):
                    raise RuntimeError(f"ComfyUI returned a non-image content type: {mime_type}")
                return image_response.content, mime_type
            time.sleep(0.5)

        raise TimeoutError(
            f"ComfyUI did not finish this image within {self.comfyui_timeout_seconds:g} seconds. "
            "Check the ComfyUI queue and increase IMAGE_GEN_COMFYUI_TIMEOUT_SECONDS if appropriate."
        )
    
    def _call_model_openrouter(self, prompt: str, reference_images: List[dict]) -> tuple:
        """Call the image generation model with retry logic."""
        logger = logging.getLogger(__name__)
        content = [{"type": "text", "text": prompt}]
        
        # Add each image with figure_id and caption label
        for img in reference_images:
            if img.get("base64") and img.get("mime_type"):
                fig_id = img.get("figure_id", "Figure")
                caption = img.get("caption", "")
                label = f"[{fig_id}]: {caption}" if caption else f"[{fig_id}]"
                content.append({"type": "text", "text": label})
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{img['mime_type']};base64,{img['base64']}"}
                })
        
        # Retry logic for API calls
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Calling image generation API (attempt {attempt + 1}/{max_retries})...")
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": content}],
                    extra_body={"modalities": ["image", "text"]}
                )
                
                # Check if response is valid
                if response is None:
                    error_msg = "API returned None response - possible rate limit or API error"
                    logger.warning(f"{error_msg} (attempt {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay * (attempt + 1))
                        continue
                    raise RuntimeError(error_msg)
                
                if not hasattr(response, 'choices') or not response.choices:
                    error_msg = f"API response has no choices: {response}"
                    logger.warning(f"{error_msg} (attempt {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay * (attempt + 1))
                        continue
                    raise RuntimeError(error_msg)
                
                message = response.choices[0].message
                if hasattr(message, 'images') and message.images:
                    image_url = message.images[0]['image_url']['url']
                    if image_url.startswith('data:'):
                        header, base64_data = image_url.split(',', 1)
                        mime_type = header.split(':')[1].split(';')[0]
                        logger.info("Image generation successful")
                        return base64.b64decode(base64_data), mime_type
                
                error_msg = "Image generation failed - no images in response"
                logger.warning(f"{error_msg} (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                raise RuntimeError(error_msg)
                
            except Exception as e:
                logger.error(f"Error in API call (attempt {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                raise
        
        raise RuntimeError("Image generation failed after all retry attempts")
    
    def _call_model_google(self, prompt: str, reference_images: List[dict]) -> tuple:
        """Call the official Google Gemini API for image generation."""
        logger = logging.getLogger(__name__)
        max_retries = 3
        retry_delay = 2  # seconds
        
        model_name = self.model if self.model.startswith("models/") else f"models/{self.model}"
        url = f"{self.google_api_base_url}/{model_name}:generateContent"
        
        model_key = model_name.split("/", 1)[-1]
        image_capable_prefixes = (
            "gemini-2.5-flash-image",
            "gemini-3-pro-image",
            "gemini-3.1-flash-image",
            "gemini-3.1-flash-lite-image",
            "gemini-nano-banana-2.1",
        )
        if not model_key.startswith(image_capable_prefixes):
            raise ValueError(
                f"Model '{model_name}' is not a recognized Google image-generation model. "
                "Use an image model such as 'models/gemini-2.5-flash-image' or "
                "'models/gemini-3-pro-image'."
            )
        
        # Compose prompt parts with optional inline reference images
        parts = [{"text": prompt}]
        for img in reference_images:
            if img.get("base64") and img.get("mime_type"):
                fig_id = img.get("figure_id", "Figure")
                caption = img.get("caption", "")
                label = f"[{fig_id}]: {caption}" if caption else f"[{fig_id}]"
                parts.append({"text": label})
                parts.append({
                    "inlineData": {
                        "mimeType": img["mime_type"],
                        "data": img["base64"],
                    }
                })
        
        payload = {
            "contents": [{"role": "user", "parts": parts}],
            # Google image models negotiate image output through modalities.
            # responseMimeType only accepts text-like formats on this endpoint.
            "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]},
        }
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Calling Google Gemini image API (attempt {attempt + 1}/{max_retries})...")
                response = requests.post(
                    url,
                    headers={"x-goog-api-key": self.api_key},
                    json=payload,
                    timeout=60,
                )
                
                if response.status_code >= 400:
                    if response.status_code == 429:
                        raise RuntimeError(
                            "Google API returned HTTP 429 (image quota/rate limit). "
                            "Check the project's AI Studio rate limits/billing before retrying."
                        )
                    # Server errors may be transient; client errors will not improve on retry.
                    retryable = response.status_code >= 500
                    if retryable and attempt < max_retries - 1:
                        logger.warning("Google API returned HTTP %s; retrying", response.status_code)
                        time.sleep(retry_delay * (attempt + 1))
                        continue
                    detail = response.text[:300]
                    raise RuntimeError(
                        f"Google API returned HTTP {response.status_code}: {detail}"
                    )
                
                data = response.json()
                candidates = data.get("candidates", [])
                if not candidates:
                    error_msg = "Google API response has no candidates"
                    logger.warning(f"{error_msg} (attempt {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay * (attempt + 1))
                        continue
                    raise RuntimeError(error_msg)
                
                parts = candidates[0].get("content", {}).get("parts", [])
                for part in parts:
                    inline = part.get("inlineData")
                    if inline and inline.get("data"):
                        mime_type = inline.get("mimeType") or self.response_mime_type
                        logger.info("Image generation successful (Google Gemini)")
                        return base64.b64decode(inline["data"]), mime_type
                    
                    text_data = part.get("text")
                    if text_data:
                        try:
                            decoded = base64.b64decode(text_data, validate=True)
                            logger.info("Image generation successful (Google Gemini, text base64 payload)")
                            return decoded, self.response_mime_type
                        except Exception:
                            continue
                
                error_msg = "Image generation failed - no image payload in response"
                logger.warning(f"{error_msg} (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                raise RuntimeError(error_msg)
            
            except Exception as e:
                if isinstance(e, RuntimeError) and "HTTP 429 (image quota/rate limit)" in str(e):
                    logger.error(str(e))
                    raise
                logger.error(f"Error in Google API call (attempt {attempt + 1}/{max_retries}): {str(e)}")
                if isinstance(e, RuntimeError) and str(e).startswith("Google API returned HTTP "):
                    raise
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                raise
        
        raise RuntimeError("Image generation failed after all retry attempts")


def save_images_as_pdf(images: List[GeneratedImage], output_path: str):
    """
    Save generated images as a single PDF file.
    
    Args:
        images: List of GeneratedImage from ImageGenerator.generate()
        output_path: Output PDF file path
    """
    from PIL import Image
    import io
    
    pdf_images = []
    
    for img in images:
        # Load image from bytes
        pil_img = Image.open(io.BytesIO(img.image_data))
        
        # Convert RGBA to RGB (PDF doesn't support alpha)
        if pil_img.mode == 'RGBA':
            pil_img = pil_img.convert('RGB')
        elif pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
        
        pdf_images.append(pil_img)
    
    if pdf_images:
        # Save first image and append the rest
        pdf_images[0].save(
            output_path,
            save_all=True,
            append_images=pdf_images[1:] if len(pdf_images) > 1 else [],
            resolution=100.0,
        )
        print(f"PDF saved: {output_path}")
