function cr3_to_png_16bit(input_path, output_path)
    % CR3_TO_PNG_16BIT Converts a Canon CR3 RAW file to a 16-bit linear PNG
    % Note: Natively requires MATLAB R2021a or newer.
    
    % 1. Verify the input file exists
    if ~isfile(input_path)
        error('Input file does not exist: %s', input_path);
    end
    
    fprintf('Reading %s and demosaicing to linear 16-bit RGB...\n', input_path);
    
    try
        % ==========================================
        % NATIVE LIBRAW DECODING (R2021a+)
        % This single function matches your rawpy configuration exactly:
        % 
        % 'ColorSpace', 'camera'             -> gamma=(1,1) (Keeps data strictly linear)
        % 'ApplyContrastStretch', false      -> no_auto_bright=True
        % 'WhiteBalanceMultipliers', 'AsTaken'-> use_camera_wb=True
        % 'BitsPerSample', 16                -> output_bps=16
        % ==========================================
        rgb_16bit = raw2rgb(input_path, ...
            'ColorSpace', 'camera', ...
            'ApplyContrastStretch', false, ...
            'WhiteBalanceMultipliers', 'AsTaken', ...
            'BitsPerSample', 16);
            
    catch ME
        error('Failed to read CR3 file. Ensure you have MATLAB R2021a+ and the Image Processing Toolbox. Error: %s', ME.message);
    end
    
    fprintf('Saving as 16-bit mathematically linear PNG...\n');
    
    % 2. Save the array to disk
    % MATLAB's imwrite natively handles uint16 matrices and saves them as 
    % lossless 16-bit PNGs automatically when the BitDepth is specified.
    imwrite(rgb_16bit, output_path, 'png', 'BitDepth', 16);
    
    fprintf('SUCCESS! Saved: %s\n', output_path);
end