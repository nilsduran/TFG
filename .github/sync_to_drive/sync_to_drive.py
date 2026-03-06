import os
import json
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload


# Carrega les credencials i el refresh token
creds = Credentials(
    None,
    refresh_token=os.getenv("REFRESH_TOKEN"),
    client_id=json.loads(os.getenv("CREDENTIALS"))["installed"]["client_id"],
    client_secret=json.loads(os.getenv("CREDENTIALS"))["installed"]["client_secret"],
    token_uri=json.loads(os.getenv("CREDENTIALS"))["installed"]["token_uri"],
)

service = build("drive", "v3", credentials=creds)
folder_id = os.getenv("FOLDER_ID")

# Carrega la configuració
# Aquesta ruta relativa ara funcionarà perquè som a l'arrel
with open(".github/sync_to_drive/config.json", "r") as f:
    config = json.load(f)

allowed_extensions = config["extensions"]
specific_files = config["files"]
ignored_files = config["ignored_files"]

# Crea un conjunt amb els fitxers específics per comparar ràpidament
specific_files_set = set(specific_files)
ignored_files_set = set(ignored_files)

# Sincronitza els fitxers
# os.walk(".") començarà a l'arrel del projecte
for root, dirs, files in os.walk("."):
    # Ignora la carpeta .github i altres fitxers del sistema
    if ".github" in root or ".git" in root:
        continue
    for file in files:
        # Ignora els fitxers a la llista d'ignorats
        if file in ignored_files_set:
            continue

        file_path = os.path.join(root, file)
        relative_path = os.path.relpath(file_path, ".")

        # Comprova si el fitxer està a la llista de fitxers específics
        if relative_path in specific_files_set:
            upload = True
        else:
            # Si no, comprova si té una extensió permesa
            upload = any(file.endswith(ext) for ext in allowed_extensions)

        if upload:
            # Afegim un print per veure el progrés als logs de GitHub
            print(f"Pujant: {file}")
            query = f"name='{file}' and '{folder_id}' in parents and trashed=false"
            response = service.files().list(q=query, spaces="drive").execute()
            files_in_drive = response.get("files", [])

            if files_in_drive:
                # Actualitza el primer fitxer trobat (Drive manté l'historial de versions automàticament)
                file_id = files_in_drive[0]["id"]
                media = MediaFileUpload(file_path, resumable=True)
                service.files().update(fileId=file_id, media_body=media).execute()
                # Mou els duplicats a la paperera (no s'esborren permanentment)
                for duplicate in files_in_drive[1:]:
                    service.files().update(fileId=duplicate["id"], body={"trashed": True}).execute()
                    print(f"  Duplicat mogut a la paperera: {file}")
            else:
                file_metadata = {"name": file, "parents": [folder_id]}
                media = MediaFileUpload(file_path, resumable=True)
                service.files().create(body=file_metadata, media_body=media, fields="id").execute()

print("Sincronització completada!")
