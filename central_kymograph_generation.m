function [binary_image] = central_kymograph_generation(segmentation_file_path, image_sequence_dir)
    % Load Segmentation 
    if exist(segmentation_file_path, 'file') ~= 2
        error('File not found: %s', segmentation_file_path);
    end
    % Load the image
    img = imread(segmentation_file_path);
    % Convert to grayscale if it's not
    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    % Convert to binary (optional, based on your requirement)
    binary_image = imbinarize(img);
    
    % Skeletonize Segmentation
    bw_skel = bwmorph(binary_image, 'skel', inf);
    bw_skel_clean = bwmorph(bw_skel, 'spur', 2);
    disp(size(bw_skel_clean));

    %Skeletonised Segmentation
    imwrite(bw_skel_clean, '/Users/lohithkonathala/iib_project/skeletonized_image.png');

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

    % Find branch points
    branch_points = bwmorph(bw_skel_clean,'branchpoints');
    edtimage = 2 * bwdist(~binary_image);

    %diameterImage has the diameter value on all points on skeleton
    diameterImage = edtimage .* double(bw_skel_clean);

    branch_points_dilate = imdilate(branch_points, strel('disk',1));

    end_points = bwmorph(bw_skel_clean,'endpoints');
    discon = bw_skel_clean & ~branch_points_dilate;

    % Read the first image in the sequence
    first_img = imread(sprintf('%s/00000.pgm', image_sequence_dir));

    % Ensure the first image is in grayscale for visualization
    if size(first_img, 3) == 3
        first_img = rgb2gray(first_img);
    end

    % Convert the first image to an RGB image
    first_img_rgb = repmat(first_img, 1, 1, 3);

    % Overlay the modified skeleton on the first image
    overlay_color = [255, 0, 0]; % Red color for overlay

    for k = 1:3
        first_img_rgb(:,:,k) = first_img_rgb(:,:,k) .* uint8(~discon) + uint8(discon) * overlay_color(k);
    end


    % Reset first image
    first_img_rgb = repmat(first_img, 1, 1, 3);

    % Parameters
    min_length = 100; % Minimum size threshold

    % Remove objects smaller than the minimum size
    discon_mini = bwareaopen(discon, min_length, 8);
    labeled = bwlabel(discon_mini,8);


    %%%% ISOLATED SEGMENTS OVERLAY (ANNOTATED) %%%%

    stats = regionprops(labeled,'Centroid','Perimeter');
    pathlength = [stats.Perimeter]/2; % Calculate the path lengths

    % Sort the segments by length in descending order and get their ranks
    [sortedLengths, sortedIndices] = sort(pathlength, 'descend');

    % Create the RGB image for the labeled vessel segments
    labeled_rgb = label2rgb(imdilate(labeled, strel('disk',1)), jet(max(labeled(:))), 'k', 'shuffle');

    % Overlay the labeled vessel segments on the first image
    overlayed_image = imadd(first_img_rgb, uint8(labeled_rgb));

    % Display the overlayed image with annotations
    figure;
    imshow(overlayed_image);
    hold on;

    stats = regionprops(labeled, 'Centroid', 'Perimeter');

    % Add length and rank annotations to the figure
    for ii = 1:numel(stats)
        % Find the rank of the current segment
        rank = find(sortedIndices == ii);
        
        % Create the annotation text (length and rank)
        annotationText = sprintf('* %0.1f pixels, Rank: %d', pathlength(ii), rank);

        % Add the text to the image
        text(stats(ii).Centroid(1), stats(ii).Centroid(2), annotationText, ...
            'FontUnits', 'normalized', 'Color', [0 0 0]);
    end
    title("Isolated Vessel Segments");
    hold off;

    % Use uiwait to keep the figure open until manually closed
    uiwait(gcf);

    % finds and counts the connected components CC in the binary image
    CC = bwconncomp(discon_mini);
    all_list = CC.PixelIdxList;
    [~,all_list_sorted] = sort(cellfun(@length,all_list));

    %%%%%%% CHOOSE VESSEL %%%%%%
    n = 9;

    index = sortedIndices(n);  % Get the index of the nth longest segment

    % Now find the actual segment using this index
    target_order = find_order(CC.PixelIdxList{index}, size(binary_image));

    % Create a binary image of the selected segment
    selected_segment = false(size(binary_image));  % Initialize with all zeros
    selected_segment(target_order) = true;  % Set the pixels of the target order to true

    % Convert the skeleton to an RGB image
    skeleton_rgb = repmat(uint8(bw_skel_clean) * 255, 1, 1, 3);  % Convert to uint8 and replicate across 3 channels

    % Highlight the selected segment in red
    selected_segment_uint8 = uint8(selected_segment) * 255;  % Convert to uint8
    skeleton_rgb(:,:,1) = skeleton_rgb(:,:,1) + selected_segment_uint8;  % Add red to the selected segment
    skeleton_rgb(:,:,2) = skeleton_rgb(:,:,2) - selected_segment_uint8;  % Remove green from the selected segment
    skeleton_rgb(:,:,3) = skeleton_rgb(:,:,3) - selected_segment_uint8;  % Remove blue from the selected segment

    %%%%%%% VESSEL SEGMENT OVERLAID ON SKELETON %%%%
    figure;
    imshow(skeleton_rgb);
    title('Vessel Segment overlaid on Skeletonized Image');
    uiwait(gcf);

    %%%% STI SETTINGS %%%%%
    starting = 1;
    N = 100;
    ending = N - starting + 1;

    index = sortedIndices(n);  % Get the index of the nth longest segment

    % Now find the actual segment using this index
    target_order = find_order(CC.PixelIdxList{index}, size(binary_image));

    % place holder empty image
    STI = zeros(length(target_order),N);

    for ii = starting:ending
        % write the img_name in format of pictures in target repository
        img_name = sprintf('%s%05d.pgm', image_sequence_dir, ii);

        A = imread(img_name);

        % Create a blank image with the same size as A
        A_segment = zeros(size(A));
        
        % Copy the nth vessel segment pixels into the blank image
        A_segment(target_order) = A(target_order);

        % Assuming 'A' is the current frame and 'target_order' contains the indices of the vessel segment
        A_colored = repmat(A, [1, 1, 3]);  % Convert grayscale frame to RGB
        A_colored(target_order) = 255;  % Set the segmented area to white (or any distinct color)
        %figure, imshow(A_colored);


        % Extract the pixel values of the nth vessel segment
        value = A(target_order);
        
        % Populate the STI
        STI(:, ii-starting+1) = value;
    end

% SEGMENT OF INTEREST %%%
imwrite(uint8(A_segment), '/Users/lohithkonathala/iib_project/vessel_segment.png');

%%%%%% SAVE KYMOGRAPH %%%%
minSTI = min(STI(:));
maxSTI = max(STI(:));
scaledSTI = (STI - minSTI) / (maxSTI - minSTI);

if isinteger(STI)
    scaledSTI = im2uint8(scaledSTI); % Convert to uint8 or use im2uint16 for uint16, etc.
end
imwrite(scaledSTI, '/Users/lohithkonathala/iib_project/central_axis_kymograph.png');
disp(['Size of STI image: ', num2str(size(STI))]);
figure;
imshow(STI, []);
uiwait(gcf);

end
