import streamlit as st
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
import time
import os
import io
import zipfile
from typing import List

from face_engine import FaceEngine
from database import FaceDatabase
from utils import get_image_files, copy_folder, export_selection

# ─── Page Config ───
st.set_page_config(
    page_title="Face Photo Manager",
    page_icon="📸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───
st.markdown("""
<style>
    .face-thumb {
        border: 3px solid #444;
        border-radius: 10px;
        padding: 2px;
        transition: border-color 0.2s;
    }
    .face-thumb:hover {
        border-color: #ff4b4b;
    }
    .stProgress .st-bo {
        background-color: #ff4b4b;
    }
    div[data-testid="stHorizontalBlock"] {
        flex-wrap: wrap;
    }
    .tag-badge {
        background-color: #ff4b4b;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.85em;
        font-weight: 500;
    }
    .stat-card {
        background: linear-gradient(135deg, #1e1e2e, #2d2d44);
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #3d3d5c;
    }
    .stat-card h2 { margin: 0; color: #ff4b4b; }
    .stat-card p { margin: 5px 0 0 0; color: #aaa; }
</style>
""", unsafe_allow_html=True)


# ─── Initialize ───
@st.cache_resource
def get_face_engine():
    return FaceEngine(det_thresh=0.5)


@st.cache_resource
def get_database():
    return FaceDatabase("data/faces.db")


engine = get_face_engine()
db = get_database()


# ─── Sidebar ───
def render_sidebar():
    with st.sidebar:
        st.title("📸 Face Photo Manager")
        st.markdown("---")

        page = st.radio(
            "Navigation",
            ["🏠 Home", "📂 Import Photos", "🔍 Process Faces", "🏷️ Tag People", "🖼️ Browse & Select"],
            label_visibility="collapsed"
        )

        st.markdown("---")
        st.markdown("### Current Project")
        projects = db.get_all_projects()
        if projects:
            project_names = [p["name"] for p in projects]
            selected = st.selectbox("Select project", project_names)
            st.session_state["current_project"] = selected
        else:
            st.info("No projects yet. Import photos to start.")
            st.session_state["current_project"] = None

        return page


# ─── Home Page ───
def page_home():
    st.title("🏠 Welcome to Face Photo Manager")
    st.markdown("""
    ### Workflow:
    1. **📂 Import Photos** — Select a folder with photos to create a working copy
    2. **🔍 Process Faces** — Detect faces and cluster them automatically
    3. **🏷️ Tag People** — Name the detected face clusters
    4. **🖼️ Browse & Select** — Filter and export photos by person
    """)

    project_name = st.session_state.get("current_project")
    if project_name:
        project = db.get_project(project_name)
        if project:
            images = db.get_all_images(project["id"])
            faces = db.get_faces_for_project(project["id"])
            clusters = db.get_cluster_face_count(project["id"])
            tags = db.get_person_tags(project["id"])

            cols = st.columns(4)
            with cols[0]:
                st.markdown(f"""<div class="stat-card">
                    <h2>{len(images)}</h2><p>Photos</p></div>""",
                    unsafe_allow_html=True)
            with cols[1]:
                st.markdown(f"""<div class="stat-card">
                    <h2>{len(faces)}</h2><p>Faces Detected</p></div>""",
                    unsafe_allow_html=True)
            with cols[2]:
                st.markdown(f"""<div class="stat-card">
                    <h2>{len(clusters)}</h2><p>Clusters</p></div>""",
                    unsafe_allow_html=True)
            with cols[3]:
                st.markdown(f"""<div class="stat-card">
                    <h2>{len(tags)}</h2><p>Tagged People</p></div>""",
                    unsafe_allow_html=True)
    else:
        st.info("👈 Start by importing a photo folder from the sidebar.")


# ─── Import Photos ───
def page_import():
    st.title("📂 Import Photos")

    col1, col2 = st.columns(2)
    with col1:
        source_folder = st.text_input(
            "Source folder path",
            placeholder="/path/to/your/photos",
            help="Path to the folder containing photos"
        )
    with col2:
        project_name = st.text_input(
            "Project name",
            placeholder="My Event Photos",
            help="A unique name for this project"
        )

    work_base = st.text_input(
        "Working directory",
        value="data/projects",
        help="Base directory where working copies are stored"
    )

    copy_files = st.checkbox("Create a working copy of the folder", value=True,
                             help="Recommended: work on a copy to keep originals safe")

    if st.button("🚀 Import", type="primary", use_container_width=True):
        if not source_folder or not project_name:
            st.error("Please provide both source folder and project name.")
            return

        source = Path(source_folder)
        if not source.exists():
            st.error(f"Folder not found: {source_folder}")
            return

        images = get_image_files(source_folder)
        if not images:
            st.error("No image files found in the folder.")
            return

        work_folder = str(Path(work_base) / project_name)

        with st.spinner("Importing photos..."):
            if copy_files:
                progress_bar = st.progress(0, text="Copying files...")
                try:
                    copy_folder(source_folder, work_folder, overwrite=False)
                    progress_bar.progress(50, text="Registering images...")
                except Exception as e:
                    st.error(f"Error copying: {e}")
                    return
            else:
                work_folder = source_folder
                progress_bar = st.progress(0, text="Registering images...")

            project_id = db.create_project(project_name, source_folder, work_folder)
            work_images = get_image_files(work_folder)

            for i, img_path in enumerate(work_images):
                try:
                    pil_img = Image.open(img_path)
                    w, h = pil_img.size
                except Exception:
                    w, h = 0, 0
                db.add_image(project_id, img_path.name, str(img_path), w, h)
                progress_bar.progress(
                    50 + int(50 * (i + 1) / len(work_images)),
                    text=f"Registering {i+1}/{len(work_images)}..."
                )

            progress_bar.progress(100, text="Done!")

        st.success(f"✅ Imported {len(work_images)} photos into project '{project_name}'")
        st.session_state["current_project"] = project_name
        st.balloons()


# ─── Process Faces ───
def page_process():
    st.title("🔍 Process Faces")

    project_name = st.session_state.get("current_project")
    if not project_name:
        st.warning("Please select or create a project first.")
        return

    project = db.get_project(project_name)
    if not project:
        st.error("Project not found.")
        return

    all_images = db.get_all_images(project["id"])
    unprocessed = db.get_unprocessed_images(project["id"])

    st.info(f"📊 {len(all_images)} total images, {len(unprocessed)} unprocessed")

    col1, col2, col3 = st.columns(3)
    with col1:
        det_threshold = st.slider("Detection confidence", 0.1, 0.9, 0.5, 0.05)
    with col2:
        cluster_eps = st.slider("Clustering sensitivity", 0.3, 1.0, 0.65, 0.05,
                                help="Lower = stricter matching, more clusters")
    with col3:
        min_cluster_size = st.slider("Min faces per cluster", 1, 10, 2)

    col_a, col_b = st.columns(2)
    with col_a:
        process_all = st.button("🔄 Reprocess All", use_container_width=True)
    with col_b:
        process_new = st.button("▶️ Process New Only", type="primary", use_container_width=True)

    if process_all:
        db.delete_project_faces(project["id"])
        images_to_process = all_images
    elif process_new:
        images_to_process = unprocessed
    else:
        # Show current state
        faces = db.get_faces_for_project(project["id"])
        if faces:
            st.markdown(f"### Current Results: {len(faces)} faces detected")
        return

    if not images_to_process:
        st.success("All images already processed!")
        return

    # Detection phase
    st.markdown("### Phase 1: Face Detection")
    progress = st.progress(0)
    status = st.empty()

    engine_local = FaceEngine(det_thresh=det_threshold)
    total = len(images_to_process)

    for i, img_data in enumerate(images_to_process):
        status.text(f"Processing {img_data['filename']} ({i+1}/{total})")
        try:
            detected = engine_local.detect_faces(img_data["filepath"])
            for face_data in detected:
                db.add_face(
                    image_id=img_data["id"],
                    project_id=project["id"],
                    bbox=face_data["bbox"],
                    embedding=face_data["embedding"],
                    confidence=face_data["confidence"]
                )
            db.mark_image_processed(img_data["id"])
        except Exception as e:
            st.warning(f"Error processing {img_data['filename']}: {e}")

        progress.progress((i + 1) / total)

    status.text("Detection complete!")

    # Clustering phase
    st.markdown("### Phase 2: Face Clustering")
    all_faces = db.get_faces_for_project(project["id"])

    if not all_faces:
        st.warning("No faces detected in any image.")
        return

    with st.spinner("Clustering faces..."):
        embeddings = np.array([f["embedding"] for f in all_faces])
        face_ids = [f["id"] for f in all_faces]

        labels = FaceEngine.cluster_faces(embeddings, eps=cluster_eps,
                                          min_samples=min_cluster_size)
        db.update_face_clusters(face_ids, labels.tolist())

        # Find representatives
        representatives = FaceEngine.find_representative_faces(embeddings, labels)

        # Initialize tags
        unique_clusters = np.unique(labels)
        for cluster_id in unique_clusters:
            cluster_id = int(cluster_id)
            rep_idx = representatives.get(cluster_id, 0)
            rep_face_id = face_ids[rep_idx] if rep_idx < len(face_ids) else None
            db.set_person_tag(project["id"], cluster_id, f"Person {cluster_id + 1}",
                              rep_face_id)

    st.success(f"✅ Found {len(all_faces)} faces in {len(unique_clusters)} clusters!")
    st.balloons()


# ─── Tag People ───
def page_tag():
    st.title("🏷️ Tag People")

    project_name = st.session_state.get("current_project")
    if not project_name:
        st.warning("Please select or create a project first.")
        return

    project = db.get_project(project_name)
    if not project:
        return

    faces = db.get_faces_for_project(project["id"])
    if not faces:
        st.warning("No faces found. Process photos first.")
        return

    tags = db.get_person_tags(project["id"])
    cluster_counts = db.get_cluster_face_count(project["id"])

    # Sort: renamed first → face count desc → cluster ID asc
    def _cluster_sort_key_tag(item):
        cid, cnt = item
        tag = tags.get(cid, {}).get("tag", f"Person {cid + 1}")
        is_default = tag == f"Person {cid + 1}"
        return (1 if is_default else 0, -cnt, cid)

    sorted_clusters = sorted(cluster_counts.items(), key=_cluster_sort_key_tag)

    st.markdown(f"### {len(sorted_clusters)} people detected")

    # Merge tool
    with st.expander("🔗 Merge Clusters"):
        st.markdown("Merge two clusters that are the same person")
        merge_cols = st.columns(3)
        cluster_options = {
            f"{tags.get(cid, {}).get('tag', f'Person {cid}')} ({cnt} faces)": cid
            for cid, cnt in sorted_clusters
        }
        with merge_cols[0]:
            src_label = st.selectbox("Source (merge from)", list(cluster_options.keys()),
                                     key="merge_src")
        with merge_cols[1]:
            tgt_label = st.selectbox("Target (merge into)", list(cluster_options.keys()),
                                     key="merge_tgt")
        with merge_cols[2]:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔗 Merge", use_container_width=True):
                src_cid = cluster_options[src_label]
                tgt_cid = cluster_options[tgt_label]
                if src_cid != tgt_cid:
                    db.merge_clusters(project["id"], src_cid, tgt_cid)
                    st.success("Clusters merged!")
                    st.rerun()

    st.markdown("---")

    # Display clusters with thumbnails and tag inputs
    for cluster_id, count in sorted_clusters:
        tag_info = tags.get(cluster_id, {})
        current_tag = tag_info.get("tag", f"Person {cluster_id}")

        with st.container():
            cols = st.columns([1, 2, 1])
            with cols[0]:
                st.markdown(f"**Cluster #{cluster_id}** — {count} faces")

            # Show representative face thumbnails
            cluster_faces = [f for f in faces if f["cluster_id"] == cluster_id]
            sample_faces = cluster_faces[:8]  # Show up to 8 samples

            thumb_cols = st.columns(min(len(sample_faces), 8))
            for j, face_data in enumerate(sample_faces):
                with thumb_cols[j]:
                    img_data = db.get_all_images(project["id"])
                    img_map = {img["id"]: img for img in img_data}
                    if face_data["image_id"] in img_map:
                        img_info = img_map[face_data["image_id"]]
                        thumb = engine.extract_face_thumbnail(
                            img_info["filepath"],
                            (face_data["bbox_x1"], face_data["bbox_y1"],
                             face_data["bbox_x2"], face_data["bbox_y2"])
                        )
                        if thumb is not None:
                            st.image(thumb, use_container_width=True)

            tag_cols = st.columns([3, 1])
            with tag_cols[0]:
                new_tag = st.text_input(
                    f"Name for cluster {cluster_id}",
                    value=current_tag,
                    key=f"tag_{cluster_id}",
                    label_visibility="collapsed",
                    placeholder="Enter person's name..."
                )
            with tag_cols[1]:
                if st.button("💾 Save", key=f"save_{cluster_id}", use_container_width=True):
                    db.set_person_tag(project["id"], cluster_id, new_tag)
                    st.success(f"Tagged as '{new_tag}'")
                    st.rerun()

            st.markdown("---")


# ─── Browse & Select ───
def page_browse():
    st.title("🖼️ Browse & Select Photos")

    project_name = st.session_state.get("current_project")
    if not project_name:
        st.warning("Please select or create a project first.")
        return

    project = db.get_project(project_name)
    if not project:
        return

    tags = db.get_person_tags(project["id"])
    cluster_counts = db.get_cluster_face_count(project["id"])

    if not tags:
        st.warning("No tagged faces found. Process and tag photos first.")
        return

    # Filter controls
    st.markdown("### 🎯 Filter by Person")
    def _cluster_sort_key_browse(cid):
        tag = tags[cid].get("tag", f"Person {cid + 1}")
        is_default = tag == f"Person {cid + 1}"
        cnt = cluster_counts.get(cid, 0)
        return (1 if is_default else 0, -cnt, cid)

    tag_options = {}
    for cid in sorted(tags.keys(), key=_cluster_sort_key_browse):
        info = tags[cid]
        label = f"{info['tag']} ({cluster_counts.get(cid, 0)} appearances)"
        tag_options[label] = cid

    selected_labels = st.multiselect(
        "Select people to show",
        list(tag_options.keys()),
        help="Select one or more people to filter photos"
    )

    selected_clusters = [tag_options[l] for l in selected_labels]

    # Filter mode
    if len(selected_clusters) >= 1:
        filter_mode = st.radio(
            "Filter mode",
            [
                "ANY — photos with any selected person",
                "ALL — photos with all selected people",
                "ONLY — photos with exclusively the selected people",
            ],
            horizontal=True
        )
    else:
        filter_mode = "ANY — photos with any selected person"

    # Get filtered images
    selected_set = set(selected_clusters)
    if selected_clusters:
        if "ONLY" in filter_mode:
            # Start with images that have ALL selected people…
            image_sets = []
            for cid in selected_clusters:
                imgs = db.get_images_by_cluster(project["id"], cid)
                image_sets.append(set(img["id"] for img in imgs))
            common_ids = set.intersection(*image_sets) if image_sets else set()
            all_project_images = {img["id"]: img for img in db.get_all_images(project["id"])}
            # …then remove any image that also contains faces from other clusters
            filtered_images = []
            for iid in common_ids:
                if iid not in all_project_images:
                    continue
                img_faces = db.get_faces_for_image(iid)
                clusters_in_image = {f["cluster_id"] for f in img_faces if f["cluster_id"] is not None}
                if clusters_in_image <= selected_set:  # subset: no extra people
                    filtered_images.append(all_project_images[iid])
        elif "ALL" in filter_mode:
            # Images that contain ALL selected people (other people allowed)
            image_sets = []
            for cid in selected_clusters:
                imgs = db.get_images_by_cluster(project["id"], cid)
                image_sets.append(set(img["id"] for img in imgs))
            common_ids = set.intersection(*image_sets) if image_sets else set()
            all_project_images = {img["id"]: img for img in db.get_all_images(project["id"])}
            filtered_images = [all_project_images[iid] for iid in common_ids
                               if iid in all_project_images]
        else:
            filtered_images = db.get_images_by_clusters(project["id"], selected_clusters)
    else:
        filtered_images = db.get_all_images(project["id"])

    st.markdown(f"### 📷 Showing {len(filtered_images)} photos")

    # Layout controls
    cols_per_row = st.slider("Columns", 2, 8, 4)

    # Selection state
    if "selected_images" not in st.session_state:
        st.session_state["selected_images"] = set()

    # Select/Deselect all
    sel_cols = st.columns(4)
    with sel_cols[0]:
        if st.button("✅ Select All Shown", use_container_width=True):
            for img in filtered_images:
                st.session_state["selected_images"].add(img["filepath"])
            st.rerun()
    with sel_cols[1]:
        if st.button("❌ Deselect All Shown", use_container_width=True):
            for img in filtered_images:
                st.session_state["selected_images"].discard(img["filepath"])
            st.rerun()
    with sel_cols[2]:
        st.markdown(f"**{len(st.session_state['selected_images'])} selected**")

    # Pagination
    items_per_page = cols_per_row * 5  # 5 rows per page
    total_pages = max(1, (len(filtered_images) + items_per_page - 1) // items_per_page)
    page_num = st.number_input("Page", 1, total_pages, 1, key="page_num")
    start_idx = (page_num - 1) * items_per_page
    end_idx = start_idx + items_per_page
    page_images = filtered_images[start_idx:end_idx]

    # Display grid
    for row_start in range(0, len(page_images), cols_per_row):
        row_images = page_images[row_start:row_start + cols_per_row]
        cols = st.columns(cols_per_row)
        for j, img_data in enumerate(row_images):
            with cols[j]:
                filepath = img_data["filepath"]
                try:
                    pil_img = Image.open(filepath)
                    pil_img.thumbnail((400, 400))
                    st.image(pil_img, use_container_width=True)

                    # Show face tags on this image
                    img_faces = db.get_faces_for_image(img_data["id"])
                    face_tags = set()
                    for f in img_faces:
                        tag_info = tags.get(f["cluster_id"])
                        if tag_info:
                            face_tags.add(tag_info["tag"])
                    if face_tags:
                        tag_str = " · ".join(face_tags)
                        st.caption(f"👤 {tag_str}")

                    is_selected = filepath in st.session_state["selected_images"]
                    if st.checkbox(
                        "Select",
                        value=is_selected,
                        key=f"sel_{img_data['id']}_{page_num}"
                    ):
                        st.session_state["selected_images"].add(filepath)
                    else:
                        st.session_state["selected_images"].discard(filepath)

                except Exception as e:
                    st.error(f"Error: {img_data['filename']}")

    # Export section
    st.markdown("---")
    st.markdown("### 📤 Export Selected Photos")

    selected_list = list(st.session_state["selected_images"])

    if not selected_list:
        st.info("Select some photos to export.")
        return

    st.markdown(f"**{len(selected_list)} photos selected for export**")

    export_cols = st.columns(2)
    with export_cols[0]:
        export_folder = st.text_input("Export folder", value="data/exports/selection")
        if st.button("📁 Export to Folder", type="primary", use_container_width=True):
            with st.spinner("Exporting..."):
                count = export_selection(selected_list, export_folder)
            st.success(f"✅ Exported {count} photos to {export_folder}")

    with export_cols[1]:
        if st.button("📦 Download as ZIP", use_container_width=True):
            with st.spinner("Creating ZIP..."):
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                    for filepath in selected_list:
                        p = Path(filepath)
                        if p.exists():
                            zf.write(p, p.name)
                zip_buffer.seek(0)

            st.download_button(
                "⬇️ Download ZIP",
                data=zip_buffer,
                file_name=f"{project_name}_selection.zip",
                mime="application/zip",
                use_container_width=True
            )


# ─── Main App ───
def main():
    page = render_sidebar()

    if page == "🏠 Home":
        page_home()
    elif page == "📂 Import Photos":
        page_import()
    elif page == "🔍 Process Faces":
        page_process()
    elif page == "🏷️ Tag People":
        page_tag()
    elif page == "🖼️ Browse & Select":
        page_browse()


if __name__ == "__main__":
    main()