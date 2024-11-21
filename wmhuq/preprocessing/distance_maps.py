def create_ventricle_distance_map(synthseg_file, out_file):
    """
    loads the synth_seg segmentation, extracts the ventricles segmentation
    and creates a euclidian distance map from each voxel to the ventricles.
    This distance map is then saved under the name out_file
    """
    
    synthseg_img = sitk.ReadImage(synthseg_file)
    
    spacing = synthseg_img.GetSpacing()
    if not ((0.95 <= spacing[0] <= 1.05) and (0.95 <= spacing[1] <= 1.05) and (0.95 <= spacing[2] <= 1.05)):
        raise ValueError(f"image spacing must be approx (1, 1, 1) to compute distance map, not {spacing}")
    
    synthseg = sitk.GetArrayFromImage(synthseg_img)
    ventricle_seg = ((synthseg == VENTRICLE_1) | (synthseg == VENTRICLE_2) | (synthseg == VENTRICLE_INFERIOR_L) | (synthseg == VENTRICLE_INFERIOR_R)).astype(np.float32)
    distance_map = distance_transform_edt(1 - ventricle_seg)
    save_manipulated_sitk_image_array(synthseg_img, distance_map, out_file)
