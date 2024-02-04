function [binary_image] = variable_axis_kymograph_generation(translated_segment_file_path, image_sequence_dir)
    if exist(translated_segment_file_path, 'file') ~= 2
        error('File not found: %s', segmentation_file_path);
    end
    %Load Image Segment 
    img = imread(translated_segment_file_path);
    % Convert to grayscale if it's not
    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    % Convert to binary (optional, based on your requirement)
    binary_image = imbinarize(img);

    %Skeletonise
    bw_skel = bwmorph(binary_image, 'skel', inf);
    bw_skel_clean = bwmorph(bw_skel, 'spur', 2);
    disp(size(bw_skel_clean));

    % Read the first image in the sequence (for STI)
    first_img = imread(sprintf('%s/00000.pgm', image_sequence_dir));

    % Ensure the first image is in grayscale for visualization
    if size(first_img, 3) == 3
        first_img = rgb2gray(first_img);
    end

    % Convert the first image to an RGB image
    first_img_rgb = repmat(first_img, 1, 1, 3);

    % Overlay the skeleton on the first image
    overlay_color = [255, 0, 0]; % Red color for overlay
    for k = 1:3
        first_img_rgb(:,:,k) = first_img_rgb(:,:,k) .* uint8(~bw_skel_clean) + uint8(bw_skel_clean) * overlay_color(k);
    end

    %%%%% OVERLAY SKELETON %%%%
    figure;
    imshow(first_img_rgb);
    uiwait(gcf);

    %%%% CHECK ENDPOINTS %%%%

    endpoints = bwmorph(bw_skel_clean, 'endpoints');
    [y, x] = find(endpoints);

    % Check the number of endpoints found
    num_endpoints = numel(x);

    if num_endpoints == 2
        disp('Two clear endpoints detected.');
        % Proceed with your analysis
    else
        disp(['Excess number of endpoints detected: ', num2str(num_endpoints)]);
        % Handle cases where there are too few or too many endpoints
    end

    %%%% GENERATE STI %%%%
    vessel_segment_indices = find(bw_skel_clean);

    % Call the find_order function to get the ordered indices of your vessel segment
    ordered_indices = find_order(vessel_segment_indices, size(bw_skel_clean));

    % Create a binary image of the selected segment
    selected_segment = false(size(bw_skel_clean));  % Initialize with all zeros
    selected_segment(ordered_indices) = true;  % Set the pixels of the target order to true


    %%%% STI SETTINGS %%%%%
    starting = 1;
    N = 100;
    ending = N - starting + 1;

    % place holder empty image
    STI = zeros(length(ordered_indices) , N);

    for ii = starting:ending
        img_name = sprintf('%s%05d.pgm', image_sequence_dir, ii);
        
        A = imread(img_name);

        % Create a blank image with the same size as A
        A_segment = zeros(size(A));
        
        % Copy the nth vessel segment pixels into the blank image
        A_segment(ordered_indices) = A(ordered_indices);

        % Assuming 'A' is the current frame and 'ordered_indices' contains the indices of the vessel segment
        A_colored = repmat(A, [1, 1, 3]);  % Convert grayscale frame to RGB
        A_colored(ordered_indices) = 255;  % Set the segmented area to white (or any distinct color)
        %figure, imshow(A_colored);


        % Extract the pixel values of the nth vessel segment
        value = A(ordered_indices);
        
        % Populate the STI
        STI(:, ii-starting+1) = value;
    end

    %%%%%% DISPLAY STI %%%%

    minSTI = min(STI(:));
    maxSTI = max(STI(:));
    scaledSTI = (STI - minSTI) / (maxSTI - minSTI);

    if isinteger(STI)
        scaledSTI = im2uint8(scaledSTI); % Convert to uint8 or use im2uint16 for uint16, etc.
    end
    imwrite(scaledSTI, '/Users/lohithkonathala/iib_project/translated_axis_kymograph.png');
    disp(['Size of STI image: ', num2str(size(STI))]);
    figure;
    imshow(STI, []);
    uiwait(gcf);