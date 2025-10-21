import gradio as gr
import os
import json
import zipfile
import tempfile
from pathlib import Path
from typing import List, Tuple, Dict, Any
import io
from PIL import Image, ImageDraw
import pdfplumber
from google.genai import types, client
from dotenv import load_dotenv
from datetime import datetime
import requests

# Load environment variables
load_dotenv()

# Initialize Gemini client
gemini_client = client.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Webhook URL for logging
WEBHOOK_URL = "https://aiagentsedata.app.n8n.cloud/webhook/cd53b28a-8e1b-4f3e-8e76-44b066114c98"

# Master prompt from the notebook
MASTER_PROMPT = """
Your task is to function as an advanced, multi-step document processing agent. You will be given a set of images for a single case (e.g., police reports and an insurance policy). Your goal is to first extract the data from each individual document, and then perform a final cross-validation to create a summary.

---
## Part 1: Individual Document Extraction

### Instructions
1.  **Analyze each Image**: For each image provided, carefully scan it to identify all text, numbers, dates, signatures, and official seals.
2.  **Translate Content**: All text written in Arabic must be accurately translated into English.
3.  **Structure the Data**: Identify which of the five document types the image corresponds to. Populate the JSON object strictly following the type notation provided for that document.

### Rules
1.  The output for this part must be **one JSON object per image**. Do not include any explanatory text or markdown formatting.
2.  Every field listed in the JSON type notation is **required**.
3.  If a field's value cannot be found in the document, use `null` as its value.
4.  For signatures or stamps, note their presence with the string `"Signature present"` or `"Present"`.
5.  Pay meticulous attention to correctly transcribing all numbers, dates, and names.
6.  For multiple, images, attribute the JSON object to its own key while enclosing in a parent JSON object.

### JSON Object Format (Type Notation)

#### Document 1: Police Report (Form 1)
{{
  "report_header": {{ "form_number": str, "report_title": str }},
  "report_details": {{ "police_station": str, "moi_file_number": str, "accident_date": str, "report_date": str, "accident_location": str }},
  "first_party": {{
    "driver_details": {{ "name": str, "civilian_id": str, "address": str, "nationality": str, "phone_number": str }},
    "driving_license": {{ "number": str, "type": Optional[str], "expiry_date": str }},
    "vehicle_details": {{ "plate_number": str, "type": str, "year_of_manufacture": str, "owner_name": str, "owner_civilian_id": str }},
    "insurance_details": {{ "company_name": str, "policy_number": str, "expiry_date": str }},
    "statement_and_damage": {{ "statement": str, "damaged_parts": str }}
  }},
  "second_party": {{
    "driver_details": {{ "name": str, "civilian_id": str, "address": str, "nationality": str, "phone_number": str }},
    "driving_license": {{ "number": str, "type": Optional[str], "expiry_date": str }},
    "vehicle_details": {{ "plate_number": str, "type": str, "year_of_manufacture": str, "owner_name": str, "company_number": Optional[str] }},
    "insurance_details": {{ "company_name": str, "policy_number": str, "expiry_date": str }},
    "statement_and_damage": {{ "statement": str, "damaged_parts": str }}
  }},
  "additional_information": {{ "other_notes": Optional[str], "report_editor": str, "footer_code": str }}
}}

#### Document 2: Acceptance of Reconciliation (Form 2)
{{
  "issuing_authority": {{ "ministry": str, "directorate_general": str, "security_directorate": str, "police_station": str }},
  "form_details": {{ "form_number": str, "form_title": str }},
  "violator_information": {{ "name": str, "civilian_id": str, "address": str }},
  "case_details": {{ "police_station": str, "record_number": str, "violation_date": str, "violation_location": str }},
  "violation_description": {{ "text": str, "is_acknowledged": bool }},
  "violator_acknowledgement": {{ "name_written": str, "signature": str }},
  "official_seal": str,
  "footer_code": str
}}

#### Document 3: Traffic Accident Notification (Form 3)
{{
  "form_details": {{ "form_number": str, "form_title": str }},
  "report_details": {{ "accident_date": str, "accident_location": str, "moi_file_number": str, "police_station": str }},
  "involved_vehicles": [
    {{ "vehicle_index": str, "plate_number": str, "insurance_policy_number": str, "driver_name": str, "insurance_company_name": str, "insurance_expiry_date": str }}
  ],
  "authorizing_officer": {{ "title": str, "rank": str, "name": str, "signature": str }},
  "official_seal": str,
  "footer_code": str
}}

#### Document 4: Insurance Policy
{{
  "issuing_company": {{ "name": str, "name_arabic": str, "authorized_capital": str, "paid_up_capital": str, "address": str, "contact": {{ "phone": str, "fax": str }} }},
  "document_title": str,
  "policy_summary": {{ "insurance_permit_no": str, "insurance_type": str, "insurance_type_code": str, "policy_year": str, "policy_number": str }},
  "policy_period": {{ "issue_date": str, "issue_time": str, "effective_from": str, "effective_to": str, "duration": str, "transaction_type": str }},
  "policy_holder": {{ "name": str, "beneficiary": str, "address": str, "phone": Optional[str] }},
  "vehicle_details": {{ "plate_number": str, "body_type": str, "make": str, "color": str, "model": str, "year_of_manufacture": str, "fuel_type": str, "chassis_number_vin": str, "engine_number": Optional[str], "passenger_capacity": int, "licensing_purpose": str }},
  "coverage_details": {{ "liability_coverage": str, "protection_value": Optional[str], "deductible_per_accident": str }},
  "premium_details": {{ "currency": str, "insurance_amount_year_1": float, "insurance_amount_year_2": float, "insurance_amount_year_3": float, "subscription_value": float, "supervision_fees": float, "issuance_expenses": float, "endorsement_fees": float, "total": float }},
  "signature": {{ "subscriber_signature": str }},
  "footer_info": {{ "company_name": str, "qr_code_note": str, "internal_ref": str, "timestamp_codes": str }}
}}

#### Document 5: Hit and Run Report & Repair Permit
{{
  "issuing_authority": {{ "ministry": str, "sector": str, "directorate": str, "command_area": str, "police_station": str }},
  "report_details": {{ "title": str, "record_number": str, "report_date": str, "accident_date": str }},
  "vehicle_and_driver": {{ "driver_name": str, "plate_number": str, "vehicle_type": str, "chassis_number_vin": str, "insurance_company": str, "policy_number": str }},
  "damage_description": {{ "damaged_parts": List[str], "designated_workshop": Optional[str] }},
  "general_instructions": str,
  "authorizing_officer": {{ "title": str, "rank": str, "name": str, "signature": str }},
  "official_seal": str,
  "footer_info": {{ "user_code": str, "timestamp": str }}
}}

---
## Part 2: Case Validation Summary

### Validation Instructions
After extracting the JSON for each document in a case, perform a final validation. Your input will be the JSON objects from the police report(s) and the insurance policy. Your output must be a single validation JSON object.

1.  **Match Insured Party**: Compare the `policy_holder` and `vehicle_details` from the **Insurance Policy JSON** with the party details in the **Police Report JSON(s)**. A successful match is based on the driver's name and/or the vehicle's VIN.
2.  **Identify Case Type**: Determine if the case is a "Two-Party Collision" or "Hit and Run" based on the police report's title.
3.  **Determine Fault (if applicable)**: If an "Acceptance of Reconciliation" document is present, identify the party who admitted fault.
4.  **List Damages**: Record the damages for the matched insured party from the police report.
5.  **Log Discrepancies**: Explicitly compare the insurer name and policy number listed on the police report against the insurance policy document. Note any mismatches.

### Validation JSON Structure
{{
  "validation_summary": {{
    "case_details": {{
      "type": "Two-Party Collision OR Hit and Run / Unknown Party",
      "police_record_number": "str"
    }},
    "validation_status": {{
      "status": "Validated OR Mismatch OR Partial Match",
      "reason": "str (e.g., Driver name and VIN on police report match the insurance policy.)"
    }},
    "insured_party_details": {{
      "party_designation": "First Party OR Second Party OR Insured Party",
      "driver_name": "str",
      "vehicle_plate_number": "str",
      "chassis_number_vin": "str"
    }},
    "at_fault_party": {{
      "party_designation": "First Party OR Second Party OR null",
      "reason": "str (e.g., Based on Acceptance of Reconciliation form.) OR null"
    }},
    "insured_vehicle_damage": {{
      "source": "Police Report",
      "damages": [
        "str"
      ]
    }},
    "validation_notes": {{
      "discrepancies_found": [
        "str (e.g., Insurer name on police report (X) does not match policy document (Y).)"
      ]
    }}
  }}
}}
"""

# Windshield claim prompt
WINDSHIELD_CLAIM_PROMPT = f"""
<claim_processing_instructions>

  <current_date>
    {datetime.now().strftime("%Y-%m-%d")}
  </current_date>

  <role>
    You are an expert claims handler based in Kuwait, specializing in processing vehicle insurance claims.
    You have extensive familiarity with Kuwaiti civil and vehicle documentation.
  </role>

  <goal>
    Process a set of submitted documents for a vehicle damage claim. Your task is to validate the completeness
    of the evidence, extract all key data, perform a multi-document consistency analysis, and deliver your
    findings in a single, structured JSON object.
  </goal>

  <input>
    A collection of images including official documents and photographs related to the claim.
  </input>

  <instructions>
    <step number="1">
      <title>Document &amp; Evidence Checklist</title>
      <description>
        Verify the presence of each required item. The output for this step should be a simple checklist
        within the final JSON.
      </description>
      <required_items>
        <item>Civil ID (Front &amp; Back)</item>
        <item>Driver's License</item>
        <item>Vehicle Registration (Istamara)</item>
        <item>Repair Order</item>
        <item>Photo of windshield damage</item>
        <item>Photo of dashboard VIN/Chassis Number</item>
        <item>Photo of vehicle license plate</item>
        <item>Photo of windshield logo (for originality check)</item>
      </required_items>
    </step>

    <step number="2">
      <title>Data Extraction &amp; Translation</title>
      <description>
        For each available document and image, meticulously extract, transcribe, and translate all relevant information.
      </description>
      <requirements>
        <requirement>Transcribe all data fields exactly as they appear</requirement>
        <requirement>Provide an accurate English translation for all Arabic text</requirement>
        <requirement>Capture all identifiers, names, dates, vehicle specifications, and claim details</requirement>
        <requirement>For damage photos, identify the damage location using normalized bounding box coordinates</requirement>
      </requirements>
    </step>

    <step number="3">
      <title>Consistency &amp; Validation Analysis</title>
      <description>
        Conduct a comprehensive cross-referencing of all extracted data to identify consistencies or discrepancies.
      </description>
      <validation_points>
        <point>
          <name>Personal Identity</name>
          <description>Confirm that the Civil ID number, name, and date of birth are consistent across all personal documents</description>
        </point>
        <point>
          <name>Vehicle Identity</name>
          <description>Confirm that the Chassis Number (VIN), license plate number, and vehicle make/model are consistent across the Vehicle Registration, repair order, and photographs</description>
        </point>
        <point>
          <name>Claim Validity</name>
          <description>Confirm that the damage depicted in the photos directly corresponds to the authorized work listed on the Repair Order</description>
        </point>
        <point>
          <name>Component Originality</name>
          <description>Note if evidence of the windshield's originality (e.g., a manufacturer's logo) is present</description>
        </point>
        <point>
            <name>Date Validation</name>
            <description>The current date is provided to you, so there must be no validation of future dates.</description>
        </point>
      </validation_points>
    </step>
  </instructions>

  <output_format>
    <description>
      Your entire response must be a single, valid JSON object matching the structure and schema below.
    </description>

    <schema_description>
      <field name="documentChecklist">An object containing boolean (true/false) values for each required item</field>
      <field name="extractedData">A parent object containing nested objects for each document (e.g., civilId, vehicleRegistration). Each nested object should contain the key-value pairs of extracted and translated information</field>
      <field name="consistencyAnalysis">An object containing an overall summary and a list of specific validation checks, each with a status and details string</field>
    </schema_description>

    <json_schema>
{{
  "documentChecklist": {{
    "summary": "string",
    "checklist": {{
      "civilIdFront": "boolean",
      "civilIdBack": "boolean",
      "driversLicense": "boolean",
      "vehicleRegistration": "boolean",
      "repairOrder": "boolean",
      "windshieldDamagePhoto": "boolean",
      "chassisNumberPhoto": "boolean",
      "licensePlatePhoto": "boolean",
      "windshieldLogoPhoto": "boolean"
    }}
  }},
  "extractedData": {{
    "civilId": {{
      "documentTitle": "string",
      "civilIdNumber": "string",
      "nameEnglish": "string",
      "nameArabic": "string",
      "nameEnglishTranslated": "string",
      "nationality": "string",
      "sex": "string",
      "birthDate": "string",
      "expiryDate": "string",
      "address": {{
        "governorate": "string",
        "block": "string",
        "street": "string",
        "building": "string",
        "unit": "string",
        "automatedAddressNumber": "string"
      }},
      "bloodType": "string",
      "phone": "string",
      "serialNumber": "string"
    }},
    "driversLicense": {{
      "documentTitle": "string",
      "licenseNumber": "string",
      "dateOfIssue": "string",
      "dateOfExpiry": "string",
      "dateOfBirth": "string",
      "nameEnglish": "string",
      "nameArabic": "string",
      "nameEnglishTranslated": "string",
      "classCode": "string",
      "nationality": "string"
    }},
    "vehicleRegistration": {{
      "documentTitle": "string",
      "plateNumber": "string",
      "licensePurpose": "string",
      "ownerCivilId": "string",
      "ownerName": "string",
      "chassisNumber_VIN": "string",
      "make": "string",
      "model": "string",
      "yearOfManufacture": "string",
      "color": "string"
    }},
    "repairOrder": {{
      "issuer": "string",
      "documentType": "string",
      "date": "string",
      "claimNumber": "string",
      "addressedTo": "string",
      "vehicleType": "string",
      "plateNumber": "string",
      "authorizedRepair": "string",
      "policyholderCost": "string"
    }},
    "photographicEvidence": {{
      "chassisNumberFromDashboard": "string",
      "licensePlateFromVehicle": "string",
      "damageDescription": "string",
      "damageLocation": {{
        "box_2d": "array of 4 integers [ymin, xmin, ymax, xmax] - normalized to 0-1000 scale",
        "imageIndex": "integer - index of the image containing the damage"
    }},
      "originalityEvidence": "string"
    }}
  }},
  "consistencyAnalysis": {{
    "overallValidationStatus": "string (VALID|INVALID|NEEDS_REVIEW)",
    "summary": "string",
    "checks": [
      {{
        "checkName": "string",
        "status": "string (CONSISTENT|INCONSISTENT|VERIFIED|NOT_VERIFIED)",
        "details": "string"
      }}
    ]
  }}
}}
    </json_schema>
  </output_format>

  <important_notes>
    <note>Ensure all extracted data is accurate and complete</note>
    <note>All Arabic text must be properly translated to English</note>
    <note>Cross-reference all documents thoroughly for consistency</note>
    <note>Flag any discrepancies or missing information clearly in the consistency analysis</note>
    <note>For damage location, provide normalized bounding box coordinates in the format [ymin, xmin, ymax, xmax] on a 0-1000 scale</note>
    <note>Include the image index (zero-based) that contains the damage in the damageLocation object</note>
  </important_notes>
</claim_processing_instructions>
"""


class DocumentProcessor:
    def __init__(self):
        self.supported_image_formats = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        self.supported_document_formats = ['.pdf']

    def convert_pdf_to_images(self, pdf_path: str) -> List[Image.Image]:
        """Convert PDF pages to PIL Images using pdfplumber"""
        try:
            images = []
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    # Convert page to image
                    page_image = page.to_image(resolution=300)
                    # Convert to PIL Image
                    pil_image = page_image.original
                    images.append(pil_image)
            return images
        except Exception as e:
            raise Exception(f"Failed to convert PDF to images: {str(e)}")

    def process_uploaded_files(self, files) -> List[Image.Image]:
        """Process uploaded files and convert them to images"""
        images = []

        if not files:
            return images

        # Handle both single file and multiple files
        if not isinstance(files, list):
            files = [files]

        for file in files:
            if file is None:
                continue

            file_path = file.name
            file_ext = Path(file_path).suffix.lower()

            if file_ext in self.supported_image_formats:
                # Load image directly
                try:
                    img = Image.open(file_path)
                    images.append(img)
                except Exception as e:
                    print(f"Error loading image {file_path}: {str(e)}")

            elif file_ext in self.supported_document_formats:
                # Convert PDF to images
                try:
                    pdf_images = self.convert_pdf_to_images(file_path)
                    images.extend(pdf_images)
                except Exception as e:
                    print(f"Error converting PDF {file_path}: {str(e)}")

        return images

    def process_zip_file(self, zip_file) -> List[Image.Image]:
        """Extract and process files from ZIP archive"""
        images = []

        if zip_file is None:
            return images

        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                with zipfile.ZipFile(zip_file.name, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)

                # Process all extracted files
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        file_ext = Path(file_path).suffix.lower()

                        if file_ext in self.supported_image_formats:
                            try:
                                img = Image.open(file_path)
                                images.append(img)
                            except Exception as e:
                                print(f"Error loading image {file_path}: {str(e)}")

                        elif file_ext in self.supported_document_formats:
                            try:
                                pdf_images = self.convert_pdf_to_images(file_path)
                                images.extend(pdf_images)
                            except Exception as e:
                                print(f"Error converting PDF {file_path}: {str(e)}")

            except Exception as e:
                print(f"Error processing ZIP file: {str(e)}")

        return images

    def images_to_gemini_parts(self, images: List[Image.Image]) -> List[types.Part]:
        """Convert PIL Images to Gemini API Parts"""
        parts = []

        for img in images:
            # Convert PIL Image to bytes
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_bytes = img_byte_arr.getvalue()

            # Create Gemini Part
            part = types.Part.from_bytes(data=img_bytes, mime_type="image/png")
            parts.append(part)

        return parts

    def process_documents_with_gemini(self, images: List[Image.Image], flow_type: str = "accident") -> str:
        """Process images with Gemini API and return JSON response"""
        if not images:
            return "No images to process"

        try:
            # Convert images to Gemini parts
            parts = self.images_to_gemini_parts(images)

            # Select prompt based on flow type
            prompt = WINDSHIELD_CLAIM_PROMPT if flow_type == "windshield" else MASTER_PROMPT

            # Configure Gemini API call
            config = types.GenerateContentConfig(
                response_mime_type="application/json",
                system_instruction=prompt,
                temperature=0.0,
            )

            # Call Gemini API with new configuration
            response = gemini_client.models.generate_content(
                model="gemini-2.5-pro",
                contents=parts,
                config=config,
            )

            return response.text

        except Exception as e:
            return f"Error processing with Gemini: {str(e)}"


# Initialize processor
processor = DocumentProcessor()


def send_webhook_log(flow_type: str, username: str, result_data: dict):
    """
    Send processing log to n8n webhook

    Args:
        flow_type: Either "windshield" or "accident"
        username: The authenticated username
        result_data: The complete JSON results from processing
    """
    try:
        payload = {
            "timestamp": datetime.now().isoformat(),
            "user": username,
            "flow_type": flow_type,
            "results": result_data
        }

        response = requests.post(
            WEBHOOK_URL,
            json=payload,
            timeout=10
        )

        if response.status_code == 200:
            print(f"Webhook sent successfully for {flow_type} flow by {username}")
        else:
            print(f"Webhook failed with status {response.status_code}: {response.text}")

    except Exception as e:
        print(f"Error sending webhook: {str(e)}")


# Windshield claim processing functions
def draw_bounding_box_on_image(image: Image.Image, box_coords: List[int]) -> Image.Image:
    """Draw bounding box on image using normalized coordinates"""
    try:
        # Convert normalized coordinates (0-1000) to actual image coordinates
        img_width, img_height = image.size
        ymin, xmin, ymax, xmax = box_coords

        # Convert to actual pixel coordinates
        x1 = int((xmin / 1000.0) * img_width)
        y1 = int((ymin / 1000.0) * img_height)
        x2 = int((xmax / 1000.0) * img_width)
        y2 = int((ymax / 1000.0) * img_height)

        # Create a copy of the image to draw on
        img_with_box = image.copy()
        draw = ImageDraw.Draw(img_with_box)

        # Draw bounding box with red color and thick lines
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

        # Add text label
        draw.text((x1, y1-20), "Damage Area", fill="red")

        return img_with_box

    except Exception as e:
        print(f"Error drawing bounding box: {e}")
        return image


def create_windshield_checklist_display(checklist_data: dict) -> str:
    """Create aesthetic display for document checklist"""
    try:
        checklist = checklist_data.get("checklist", {})
        summary = checklist_data.get("summary", "Document checklist completed")

        html = f"""
        <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin: 10px 0;">
            <h3 style="color: #2c3e50; margin-bottom: 20px; text-align: center;">📋 Document Checklist</h3>
            <p style="text-align: center; margin-bottom: 20px; color: #666;"><em>{summary}</em></p>

            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px;">
        """

        checklist_items = {
            "civilIdFront": "📋 Civil ID (Front)",
            "civilIdBack": "📋 Civil ID (Back)",
            "driversLicense": "🚗 Driver's License",
            "vehicleRegistration": "📄 Vehicle Registration (Istamara)",
            "repairOrder": "🔧 Repair Order",
            "windshieldDamagePhoto": "📸 Windshield Damage Photo",
            "chassisNumberPhoto": "🔢 Dashboard VIN/Chassis Photo",
            "licensePlatePhoto": "🚙 License Plate Photo",
            "windshieldLogoPhoto": "🏷️ Windshield Logo Photo"
        }

        for key, label in checklist_items.items():
            status = checklist.get(key, False)
            status_color = "#28a745" if status else "#dc3545"
            status_icon = "✅" if status else "❌"
            status_text = "Present" if status else "Missing"

            html += f"""
                <div style="background-color: white; padding: 15px; border-radius: 8px; border-left: 4px solid {status_color}; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <div style="display: flex; align-items: center; justify-content: space-between;">
                        <div style="display: flex; align-items: center;">
                            <span style="margin-right: 10px; color: #212529; font-size: 14px;">{label}</span>
                        </div>
                        <div style="display: flex; align-items: center;">
                            <span style="margin-right: 5px; font-size: 16px;">{status_icon}</span>
                            <span style="color: {status_color}; font-weight: bold; font-size: 14px;">{status_text}</span>
                        </div>
                    </div>
                </div>
            """

        html += """
            </div>
        </div>
        """

        return html

    except Exception as e:
        return f"<p>Error creating checklist display: {str(e)}</p>"


def create_extraction_tiles(extracted_data: dict, current_index: int = 0) -> tuple:
    """Create tile display for extracted data with navigation"""
    try:
        sections = []
        section_names = []

        # Define section order and labels
        section_mapping = {
            "civilId": "👤 Civil ID",
            "driversLicense": "🚗 Driver's License",
            "vehicleRegistration": "📄 Vehicle Registration",
            "repairOrder": "🔧 Repair Order",
            "photographicEvidence": "📸 Photographic Evidence"
        }

        for key, label in section_mapping.items():
            if key in extracted_data:
                sections.append(extracted_data[key])
                section_names.append(label)

        if not sections:
            return "<p>No extraction data available</p>", 0, 0

        # Ensure current_index is within bounds
        current_index = max(0, min(current_index, len(sections) - 1))
        current_section = sections[current_index]
        current_name = section_names[current_index]

        html = f"""
        <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin: 10px 0;">
            <div style="text-align: center; margin-bottom: 20px;">
                <h3 style="color: #2c3e50; margin-bottom: 10px;">{current_name}</h3>
                <p style="color: #666; margin: 0;">Section {current_index + 1} of {len(sections)}</p>
            </div>

            <div style="background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        """

        # Handle different section types
        if isinstance(current_section, dict):
            for field_key, field_value in current_section.items():
                if isinstance(field_value, dict):
                    # Nested object (like address or damage location)
                    if field_key == "damageLocation":
                        # Special handling for damage location
                        html += f"""
                            <div style="margin-bottom: 15px; padding: 15px; background-color: #fff3cd; border-radius: 8px; border-left: 4px solid #ffc107;">
                                <h4 style="color: #856404; margin-bottom: 8px;">📍 Damage Location</h4>
                                <div style="margin-left: 20px;">
                        """
                        for sub_key, sub_value in field_value.items():
                            if sub_key == "box_2d":
                                html += f"""
                                    <div style="margin-bottom: 5px;">
                                        <strong style="color: #212529;">Bounding Box Coordinates:</strong> <span style="color: #212529;">{sub_value if sub_value else 'N/A'}</span>
                                        <small style="color: #666; display: block; font-style: italic;">Format: [ymin, xmin, ymax, xmax] (normalized 0-1000)</small>
                                    </div>
                                """
                            else:
                                html += f"""
                                    <div style="margin-bottom: 5px;">
                                        <strong style="color: #212529;">{sub_key.replace('_', ' ').title()}:</strong> <span style="color: #212529;">{sub_value if sub_value else 'N/A'}</span>
                                    </div>
                                """
                        html += """
                                </div>
                                <div style="margin-top: 10px; padding: 8px; background-color: #d1ecf1; border-radius: 4px;">
                                    <small style="color: #0c5460;">
                                        💡 <strong>Note:</strong> The damage area is highlighted with a red bounding box in the "Damage Evidence" section.
                                    </small>
                                </div>
                            </div>
                        """
                    else:
                        # Regular nested object (like address)
                        html += f"""
                            <div style="margin-bottom: 15px;">
                                <h4 style="color: #212529; margin-bottom: 8px; text-transform: capitalize;">{field_key.replace('_', ' ')}</h4>
                                <div style="margin-left: 20px; padding: 10px; background-color: #f8f9fa; border-radius: 4px;">
                        """
                        for sub_key, sub_value in field_value.items():
                            html += f"""
                                <div style="margin-bottom: 5px;">
                                    <strong style="color: #212529;">{sub_key.replace('_', ' ').title()}:</strong> <span style="color: #212529;">{sub_value if sub_value else 'N/A'}</span>
                                </div>
                            """
                        html += "</div></div>"
                else:
                    # Regular field
                    if field_key == "damageDescription":
                        # Special styling for damage description
                        html += f"""
                            <div style="margin-bottom: 15px; padding: 12px; background-color: #f8d7da; border-radius: 8px; border-left: 4px solid #dc3545;">
                                <strong style="color: #721c24;">🔍 Damage Description:</strong>
                                <p style="margin: 8px 0 0 0; color: #721c24;">{field_value if field_value else 'N/A'}</p>
                            </div>
                        """
                    else:
                        # Regular field
                        html += f"""
                            <div style="margin-bottom: 12px; padding: 8px; border-bottom: 1px solid #e9ecef;">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <strong style="color: #212529;">{field_key.replace('_', ' ').title()}:</strong>
                                    <span style="color: #212529;">{field_value if field_value else 'N/A'}</span>
                                </div>
                            </div>
                        """

        html += """
            </div>
        </div>
        """

        return html, current_index, len(sections)

    except Exception as e:
        return f"<p>Error creating extraction tiles: {str(e)}</p>", 0, 0


def create_validation_display(consistency_analysis: dict) -> str:
    """Create display for validation results"""
    try:
        overall_status = consistency_analysis.get("overallValidationStatus", "UNKNOWN")
        summary = consistency_analysis.get("summary", "No summary available")
        checks = consistency_analysis.get("checks", [])

        # Status colors
        status_colors = {
            "VALID": "#28a745",
            "INVALID": "#dc3545",
            "NEEDS_REVIEW": "#ffc107"
        }

        status_icons = {
            "VALID": "✅",
            "INVALID": "❌",
            "NEEDS_REVIEW": "⚠️"
        }

        overall_color = status_colors.get(overall_status, "#6c757d")
        overall_icon = status_icons.get(overall_status, "❓")

        html = f"""
        <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin: 10px 0;">
            <h3 style="color: #2c3e50; margin-bottom: 20px; text-align: center;">🔍 Validation Results</h3>

            <div style="background-color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; border-left: 4px solid {overall_color};">
                <div style="display: flex; align-items: center; margin-bottom: 15px;">
                    <span style="font-size: 24px; margin-right: 10px;">{overall_icon}</span>
                    <div>
                        <h4 style="margin: 0; color: {overall_color};">Overall Status: {overall_status}</h4>
                        <p style="margin: 5px 0 0 0; color: #666;">{summary}</p>
                    </div>
                </div>
            </div>

            <h4 style="color: #2c3e50; margin-bottom: 15px;">Individual Checks:</h4>
        """

        check_status_colors = {
            "CONSISTENT": "#28a745",
            "INCONSISTENT": "#dc3545",
            "VERIFIED": "#28a745",
            "NOT_VERIFIED": "#ffc107"
        }

        check_status_icons = {
            "CONSISTENT": "✅",
            "INCONSISTENT": "❌",
            "VERIFIED": "✅",
            "NOT_VERIFIED": "⚠️"
        }

        for check in checks:
            check_name = check.get("checkName", "Unknown Check")
            check_status = check.get("status", "UNKNOWN")
            check_details = check.get("details", "No details available")

            check_color = check_status_colors.get(check_status, "#6c757d")
            check_icon = check_status_icons.get(check_status, "❓")

            html += f"""
                <div style="background-color: white; padding: 15px; border-radius: 8px; margin-bottom: 10px; border-left: 4px solid {check_color}; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <div style="display: flex; align-items: center; margin-bottom: 8px;">
                        <span style="font-size: 18px; margin-right: 10px;">{check_icon}</span>
                        <h5 style="margin: 0; color: #2c3e50;">{check_name}</h5>
                        <span style="margin-left: auto; color: {check_color}; font-weight: bold;">{check_status}</span>
                    </div>
                    <p style="margin: 0; color: #666; margin-left: 28px;">{check_details}</p>
                </div>
            """

        html += """
        </div>
        """

        return html

    except Exception as e:
        return f"<p>Error creating validation display: {str(e)}</p>"


def process_files_accident(individual_files, zip_file, request: gr.Request):
    """Processing function for accident claims"""
    try:
        all_images = []

        # Process individual files
        if individual_files:
            individual_images = processor.process_uploaded_files(individual_files)
            all_images.extend(individual_images)

        # Process ZIP file
        if zip_file:
            zip_images = processor.process_zip_file(zip_file)
            all_images.extend(zip_images)

        if not all_images:
            error_msg = "No valid images found to process"
            return error_msg, None, 0

        # Process with Gemini using accident flow
        result = processor.process_documents_with_gemini(all_images, flow_type="accident")

        # Send webhook log
        try:
            # Get username from Gradio request context
            username = request.username if request and hasattr(request, 'username') else "unknown"
            # Parse result to send as JSON
            try:
                parsed_result = json.loads(result)
                send_webhook_log("accident", username, parsed_result)
            except:
                # If result is not valid JSON, send as string
                send_webhook_log("accident", username, {"raw_result": result})
        except Exception as e:
            print(f"Error sending webhook in accident flow: {e}")

        return (
            result,                         # Complete JSON output
            all_images,                     # Image gallery
            len(all_images)                 # Image count
        )

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        return error_msg, None, 0


def process_files_windshield(individual_files, zip_file, request: gr.Request):
    """Processing function for windshield claims"""
    try:
        all_images = []

        # Process individual files
        if individual_files:
            individual_images = processor.process_uploaded_files(individual_files)
            all_images.extend(individual_images)

        # Process ZIP file
        if zip_file:
            zip_images = processor.process_zip_file(zip_file)
            all_images.extend(zip_images)

        if not all_images:
            error_msg = "No valid images found to process"
            return error_msg, "", "", error_msg, None, None, 0, "", 0, 0

        # Process with Gemini using windshield flow
        result = processor.process_documents_with_gemini(all_images, flow_type="windshield")

        # Parse windshield result
        try:
            parsed_result = json.loads(result)
        except:
            error_msg = f"Error parsing result: {result}"
            return error_msg, "", "", error_msg, None, None, 0, "", 0, 0

        # Extract sections
        document_checklist = parsed_result.get("documentChecklist", {})
        extracted_data = parsed_result.get("extractedData", {})
        consistency_analysis = parsed_result.get("consistencyAnalysis", {})

        # Create displays
        checklist_html = create_windshield_checklist_display(document_checklist)
        extraction_html, current_idx, total_sections = create_extraction_tiles(extracted_data, 0)
        validation_html = create_validation_display(consistency_analysis)

        # Handle photographic evidence with bounding box
        damage_image = None
        photographic_evidence = extracted_data.get("photographicEvidence", {})
        damage_location = photographic_evidence.get("damageLocation", {})

        if damage_location and "imageIndex" in damage_location and "box_2d" in damage_location:
            try:
                image_index = damage_location["imageIndex"]
                box_coords = damage_location["box_2d"]
                if isinstance(box_coords, list) and len(box_coords) == 4 and image_index < len(all_images):
                    damage_image = draw_bounding_box_on_image(all_images[image_index], box_coords)
            except Exception as e:
                print(f"Error processing damage location: {e}")

        # Send webhook log
        try:
            # Get username from Gradio request context
            username = request.username if request and hasattr(request, 'username') else "unknown"
            send_webhook_log("windshield", username, parsed_result)
        except Exception as e:
            print(f"Error sending webhook in windshield flow: {e}")

        return (
            checklist_html,          # Document checklist
            extraction_html,         # Extraction data
            validation_html,         # Validation results
            json.dumps(parsed_result, indent=2, ensure_ascii=False),  # Complete JSON
            all_images,              # Image gallery
            damage_image,            # Damage image with bounding box
            len(all_images),         # Image count
            result,                  # Raw result for debugging
            current_idx,             # Current extraction index
            total_sections           # Total extraction sections
        )

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        return error_msg, "", "", error_msg, None, None, 0, "", 0, 0


def navigate_extraction_tiles(extracted_data_json: str, direction: str, current_index: int):
    """Navigate through extraction tiles"""
    try:
        if not extracted_data_json:
            return "<p>No data available</p>", 0, 0

        # Parse the complete JSON to get extracted data
        parsed_result = json.loads(extracted_data_json)
        extracted_data = parsed_result.get("extractedData", {})

        # Calculate new index
        section_count = len([k for k in ["civilId", "driversLicense", "vehicleRegistration", "repairOrder", "photographicEvidence"] if k in extracted_data])

        if direction == "prev":
            new_index = max(0, current_index - 1)
        elif direction == "next":
            new_index = min(section_count - 1, current_index + 1)
        else:
            new_index = current_index

        # Create new tile display
        extraction_html, final_index, total_sections = create_extraction_tiles(extracted_data, new_index)

        return extraction_html, final_index, total_sections

    except Exception as e:
        return f"<p>Error navigating: {str(e)}</p>", current_index, 0


def create_overall_summary(documents, total_images):
    """Create an overall summary of all processed documents"""
    if not documents:
        return "No documents were processed successfully."

    summary = f"# Processing Summary\n\n"
    summary += f"**Total Images Processed:** {total_images}\n"
    summary += f"**Documents Identified:** {len(documents)}\n\n"

    summary += "## Document Overview\n\n"

    for i, doc in enumerate(documents, 1):
        summary += f"### Document {i}: {doc['type']}\n"
        summary += doc['summary']
        summary += "\n---\n\n"

    return summary


def create_individual_docs_html(documents):
    """Create HTML for individual document tabs"""
    if not documents:
        return "<p>No documents to display</p>"

    html = ""
    for i, doc in enumerate(documents, 1):
        html += f"""
        <div style="margin-bottom: 30px; border: 1px solid #ddd; padding: 20px; border-radius: 8px;">
            <h3>Document {i}: {doc['type']}</h3>
            <div style="margin: 10px 0;">
                <h4>Summary:</h4>
                <div style="background-color: #f5f5f5; padding: 10px; border-radius: 4px; white-space: pre-line;">
{doc['summary']}
                </div>
            </div>
            <div style="margin: 10px 0;">
                <h4>Complete JSON Data:</h4>
                <pre style="background-color: #f8f8f8; padding: 10px; border-radius: 4px; overflow-x: auto; font-size: 12px;">{doc['content']}</pre>
            </div>
        </div>
        """

    return html


def parse_and_format_json(json_result):
    """Parse JSON result and create formatted display for different tabs"""
    if not json_result or json_result.startswith("Error"):
        return json_result, [], ""

    try:
        # Try to parse the JSON
        parsed_json = json.loads(json_result)

        # Format the complete JSON
        formatted_complete = json.dumps(parsed_json, indent=2, ensure_ascii=False)

        # Extract individual documents
        documents = []
        validation_summary = ""

        if isinstance(parsed_json, dict):
            # Check if there's a validation summary
            if "validation_summary" in parsed_json:
                validation_summary = json.dumps(parsed_json["validation_summary"], indent=2, ensure_ascii=False)
                # Remove validation from main document parsing
                parsed_json_docs = {k: v for k, v in parsed_json.items() if k != "validation_summary"}
            else:
                parsed_json_docs = parsed_json

            # Identify document types and create structured tabs
            for key, value in parsed_json_docs.items():
                if isinstance(value, dict):
                    doc_type = identify_document_type(value)
                    documents.append({
                        "key": key,
                        "type": doc_type,
                        "content": json.dumps(value, indent=2, ensure_ascii=False),
                        "summary": create_document_summary(value, doc_type)
                    })

        return formatted_complete, documents, validation_summary

    except json.JSONDecodeError:
        return json_result, [], ""

def identify_document_type(doc_data):
    """Identify the type of document based on its structure"""
    if "report_header" in doc_data and "first_party" in doc_data:
        return "Police Report (Form 1)"
    elif "violator_information" in doc_data and "violation_description" in doc_data:
        return "Acceptance of Reconciliation (Form 2)"
    elif "involved_vehicles" in doc_data and "authorizing_officer" in doc_data:
        return "Traffic Accident Notification (Form 3)"
    elif "issuing_company" in doc_data and "policy_summary" in doc_data:
        return "Insurance Policy (Form 4)"
    elif "vehicle_and_driver" in doc_data and "damage_description" in doc_data:
        return "Hit and Run Report (Form 5)"
    else:
        return "Unknown Document Type"

def create_document_summary(doc_data, doc_type):
    """Create a human-readable summary of the document"""
    summary = f"**Document Type:** {doc_type}\n\n"

    try:
        if doc_type == "Police Report (Form 1)":
            summary += f"**Police Station:** {doc_data.get('report_details', {}).get('police_station', 'N/A')}\n"
            summary += f"**Accident Date:** {doc_data.get('report_details', {}).get('accident_date', 'N/A')}\n"
            summary += f"**Location:** {doc_data.get('report_details', {}).get('accident_location', 'N/A')}\n"
            summary += f"**File Number:** {doc_data.get('report_details', {}).get('moi_file_number', 'N/A')}\n\n"

            summary += "**First Party:**\n"
            first_party = doc_data.get('first_party', {})
            summary += f"- Name: {first_party.get('driver_details', {}).get('name', 'N/A')}\n"
            summary += f"- Vehicle: {first_party.get('vehicle_details', {}).get('plate_number', 'N/A')}\n"
            summary += f"- Damage: {first_party.get('statement_and_damage', {}).get('damaged_parts', 'N/A')}\n\n"

            summary += "**Second Party:**\n"
            second_party = doc_data.get('second_party', {})
            summary += f"- Name: {second_party.get('driver_details', {}).get('name', 'N/A')}\n"
            summary += f"- Vehicle: {second_party.get('vehicle_details', {}).get('plate_number', 'N/A')}\n"
            summary += f"- Damage: {second_party.get('statement_and_damage', {}).get('damaged_parts', 'N/A')}\n"

        elif doc_type == "Insurance Policy (Form 4)":
            summary += f"**Company:** {doc_data.get('issuing_company', {}).get('name', 'N/A')}\n"
            summary += f"**Policy Number:** {doc_data.get('policy_summary', {}).get('policy_number', 'N/A')}\n"
            summary += f"**Policy Holder:** {doc_data.get('policy_holder', {}).get('name', 'N/A')}\n"
            summary += f"**Vehicle:** {doc_data.get('vehicle_details', {}).get('make', 'N/A')} {doc_data.get('vehicle_details', {}).get('model', 'N/A')}\n"
            summary += f"**Plate Number:** {doc_data.get('vehicle_details', {}).get('plate_number', 'N/A')}\n"
            summary += f"**Effective Period:** {doc_data.get('policy_period', {}).get('effective_from', 'N/A')} to {doc_data.get('policy_period', {}).get('effective_to', 'N/A')}\n"

        elif doc_type == "Traffic Accident Notification (Form 3)":
            summary += f"**Accident Date:** {doc_data.get('report_details', {}).get('accident_date', 'N/A')}\n"
            summary += f"**Location:** {doc_data.get('report_details', {}).get('accident_location', 'N/A')}\n"
            summary += f"**File Number:** {doc_data.get('report_details', {}).get('moi_file_number', 'N/A')}\n"
            summary += f"**Police Station:** {doc_data.get('report_details', {}).get('police_station', 'N/A')}\n"

        elif doc_type == "Acceptance of Reconciliation (Form 2)":
            summary += f"**Violator:** {doc_data.get('violator_information', {}).get('name', 'N/A')}\n"
            summary += f"**Violation Date:** {doc_data.get('case_details', {}).get('violation_date', 'N/A')}\n"
            summary += f"**Location:** {doc_data.get('case_details', {}).get('violation_location', 'N/A')}\n"
            summary += f"**Record Number:** {doc_data.get('case_details', {}).get('record_number', 'N/A')}\n"

        elif doc_type == "Hit and Run Report (Form 5)":
            summary += f"**Driver:** {doc_data.get('vehicle_and_driver', {}).get('driver_name', 'N/A')}\n"
            summary += f"**Vehicle:** {doc_data.get('vehicle_and_driver', {}).get('vehicle_type', 'N/A')}\n"
            summary += f"**Plate Number:** {doc_data.get('vehicle_and_driver', {}).get('plate_number', 'N/A')}\n"
            summary += f"**Report Date:** {doc_data.get('report_details', {}).get('report_date', 'N/A')}\n"

    except Exception as e:
        summary += f"Error creating summary: {str(e)}"

    return summary


# Create Gradio interface
with gr.Blocks(title="FNOL Document Processing Flow", theme=gr.themes.Soft()) as app:
    gr.Markdown("# FNOL Document Processing Flow")
    gr.Markdown("Select your claim type and upload documents for AI-assisted processing.")

    with gr.Tabs() as main_tabs:
        # WINDSHIELD CLAIM TAB
        with gr.Tab("🚗 Windshield Claim"):
            gr.Markdown("### Windshield Damage Claim Processing")
            gr.Markdown("Upload documents and photos for windshield damage claims with advanced validation and bounding box detection.")

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### File Upload")

                    windshield_individual_files = gr.File(
                        label="Upload Individual Documents",
                        file_count="multiple",
                        file_types=[".png", ".jpg", ".jpeg", ".pdf", ".bmp", ".tiff"]
                    )

                    windshield_zip_file = gr.File(
                        label="Upload ZIP File",
                        file_types=[".zip"]
                    )

                    windshield_process_btn = gr.Button("Process Windshield Claim", variant="primary")

                    gr.Markdown("### Processing Info")
                    windshield_image_count = gr.Number(label="Total Images Processed", value=0)

                with gr.Column(scale=2):
                    gr.Markdown("### Claim Processing Results")

                    with gr.Tabs() as windshield_result_tabs:
                        with gr.Tab("📋 Document Checklist"):
                            windshield_checklist = gr.HTML()

                            with gr.Row():
                                windshield_next_btn = gr.Button("Continue to Extraction →", variant="secondary")

                        with gr.Tab("📊 Extracted Data"):
                            windshield_extraction = gr.HTML()

                            with gr.Row():
                                with gr.Column(scale=1):
                                    windshield_prev_btn = gr.Button("← Previous", variant="secondary")
                                with gr.Column(scale=2):
                                    windshield_extraction_nav = gr.Markdown("")
                                with gr.Column(scale=1):
                                    windshield_next_extraction_btn = gr.Button("Next →", variant="secondary")

                            with gr.Row():
                                with gr.Column():
                                    gr.Markdown("### Damage Evidence with Bounding Box")
                                    windshield_damage_image = gr.Image(
                                        label="Damage Location",
                                        show_label=True,
                                        height=400
                                    )

                        with gr.Tab("✅ Validation Results"):
                            windshield_validation = gr.HTML()

                        with gr.Tab("📄 Complete JSON"):
                            windshield_json_output = gr.Code(
                                label="Complete JSON Output",
                                language="json",
                                lines=20
                            )

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Processed Images Gallery")
                    windshield_image_gallery = gr.Gallery(
                        label="All Processed Images",
                        show_label=True,
                        elem_id="windshield_gallery",
                        columns=3,
                        rows=2,
                        height="auto"
                    )

            # Hidden state variables for windshield flow
            windshield_raw_result = gr.Textbox(visible=False)  # Store raw result for navigation
            windshield_current_idx = gr.Number(visible=False, value=0)  # Current extraction index
            windshield_total_sections = gr.Number(visible=False, value=0)  # Total sections

        # ACCIDENT CLAIM TAB
        with gr.Tab("🚨 Accident Claim"):
            gr.Markdown("### Traffic Accident Claim Processing")
            gr.Markdown("Upload police reports, insurance policies, and related documents for comprehensive accident claim processing.")

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### File Upload")

                    accident_individual_files = gr.File(
                        label="Upload Individual Documents",
                        file_count="multiple",
                        file_types=[".png", ".jpg", ".jpeg", ".pdf", ".bmp", ".tiff"]
                    )

                    accident_zip_file = gr.File(
                        label="Upload ZIP File",
                        file_types=[".zip"]
                    )

                    accident_process_btn = gr.Button("Process Accident Claim", variant="primary")

                    gr.Markdown("### Processing Info")
                    accident_image_count = gr.Number(label="Total Images Processed", value=0)

                with gr.Column(scale=2):
                    gr.Markdown("### Claim Processing Results")

                    accident_json_output = gr.Code(
                        label="Complete JSON Output",
                        language="json",
                        lines=20
                    )

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Processed Images Preview")
                    accident_image_gallery = gr.Gallery(
                        label="Processed Images",
                        show_label=True,
                        elem_id="accident_gallery",
                        columns=3,
                        rows=2,
                        height="auto"
                    )

    # Event handlers for Windshield Claims
    windshield_process_btn.click(
        fn=process_files_windshield,
        inputs=[windshield_individual_files, windshield_zip_file],
        outputs=[
            windshield_checklist,
            windshield_extraction,
            windshield_validation,
            windshield_json_output,
            windshield_image_gallery,
            windshield_damage_image,
            windshield_image_count,
            windshield_raw_result,
            windshield_current_idx,
            windshield_total_sections
        ]
    )

    # Navigation handlers for windshield extraction tiles
    windshield_prev_btn.click(
        fn=navigate_extraction_tiles,
        inputs=[windshield_raw_result, gr.State("prev"), windshield_current_idx],
        outputs=[windshield_extraction, windshield_current_idx, windshield_total_sections]
    )

    windshield_next_extraction_btn.click(
        fn=navigate_extraction_tiles,
        inputs=[windshield_raw_result, gr.State("next"), windshield_current_idx],
        outputs=[windshield_extraction, windshield_current_idx, windshield_total_sections]
    )

    # Tab navigation for windshield flow
    windshield_next_btn.click(
        fn=lambda: gr.Tabs.update(selected=1),
        outputs=windshield_result_tabs
    )

    # Event handlers for Accident Claims
    accident_process_btn.click(
        fn=process_files_accident,
        inputs=[accident_individual_files, accident_zip_file],
        outputs=[
            accident_json_output,
            accident_image_gallery,
            accident_image_count
        ]
    )


if __name__ == "__main__":
    # User authentication credentials
    auth_users = [
        ("admin", "password"),
        ("bader.alghunaim", "Bd2k9$mQ"),
        ("abdullah.alasfoor", "Ab7x3!pL"),
        ("mishal.almana", "Mh4s6#nR"),
        ("quteba.altoma", "Qt8w1@tK"),
        ("mohmed.allam", "Mh5y9$aM"),
        ("sampath.aravamudan", "Sp2v4!rN"),
        ("samer.aljurdi", "Sm6z8#jD"),
        ("saher.abujamous", "Sh3x7@jS"),
        ("atif.ahmad", "At9k2$hA"),
        ("stijen.venrooij", "St5m7#vR")
    ]


    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 8080)),
        ssr_mode=False,
        debug=False, 
        share=False,
        inbrowser=False,
        auth=auth_users
        )