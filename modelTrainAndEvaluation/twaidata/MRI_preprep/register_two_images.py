import SimpleITK as sitk

def register_two_images_of_different_modalities(fixed_image_path, moving_image_path):
    """
    works well for different modalities, uses a multiresolution framework. It is very slow.
    Takes a few minutes per brain. In a tuned version of the simpler registration method below.
    """
    
    # requires 32 bit float type for 
    fixed_img = sitk.ReadImage(fixed_image_path, sitk.sitkFloat32)
    moving_img = sitk.ReadImage(moving_image_path, sitk.sitkFloat32)
    
    # ensure direction and origin the same as fixed image.
    # fixing orientation essential to ensuring the images are registered
    # the right way round with one another (e.g so it doesn't morph one image
    # upside down onto another.
    moving_img.SetDirection(fixed_img.GetDirection())
    moving_img.SetOrigin(fixed_img.GetOrigin())
    
    # initial transform aligns the centres
    initial_transform = sitk.CenteredTransformInitializer(fixed_img, 
                                                      moving_img, 
                                                      sitk.Euler3DTransform(), 
                                                      sitk.CenteredTransformInitializerFilter.MOMENTS)
    
    
    registration_method = sitk.ImageRegistrationMethod()

    # Similarity metric settings. # not sure if the sampling strategy stuff
    # is even used for mean squares... but this seems to work well.
    registration_method.SetMetricAsANTSNeighborhoodCorrelation(10)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.03)

    # registration_method.SetInterpolator(sitk.sitkBSpline5)
    registration_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings.
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=150, convergenceMinimumValue=1e-6, convergenceWindowSize=8)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework.            
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # dont transform in place (useful for debugging)
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    final_transform = registration_method.Execute(fixed_img, moving_img)
    
    # compute the final transform on the image
    # to transform a binary mask change sitk.sitkLinear to sitk.sitkNearestNeighbor
    registered_img = sitk.Resample(moving_img, fixed_img, final_transform, sitk.sitkLinear, 0.0, moving_img.GetPixelID())
    
    return fixed_img, registered_img


def register_two_images(fixed_image_path, moving_image_path):
    # requires 32 bit float type for 
    fixed_img = sitk.ReadImage(fixed_image_path, sitk.sitkFloat32)
    moving_img = sitk.ReadImage(moving_image_path, sitk.sitkFloat32)
    
    # ensure direction and origin the same as fixed image.
    # fixing orientation essential to ensuring the images are registered
    # the right way round with one another (e.g so it doesn't morph one image
    # upside down onto another.
    moving_img.SetDirection(fixed_img.GetDirection())
    moving_img.SetOrigin(fixed_img.GetOrigin())
    
    # initial transform aligns the centres
    initial_transform = sitk.CenteredTransformInitializer(fixed_img, 
                                                      moving_img, 
                                                      sitk.Euler3DTransform(), 
                                                      sitk.CenteredTransformInitializerFilter.GEOMETRY)
    
    
    registration_method = sitk.ImageRegistrationMethod()

    # Similarity metric settings. # not sure if the sampling strategy stuff
    # is even used for mean squares... but this seems to work well.
    registration_method.SetMetricAsMeanSquares()
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.03)

    registration_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings.
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework.            
    # registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
    # registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
    # registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # dont transform in place (useful for debugging)
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    final_transform = registration_method.Execute(fixed_img, moving_img)
    # final_transform = initial_transform
    
    # compute the final transform on the image
    # to transform a binary mask change sitk.sitkLinear to sitk.sitkNearestNeighbor    
    registered_img = sitk.Resample(moving_img, fixed_img, final_transform, sitk.sitkLinear, 0.0, moving_img.GetPixelID())
    
    return fixed_img, registered_img