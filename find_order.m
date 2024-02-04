function order = find_order(skel_image_ids, image_size)

    skel_image = zeros(image_size);
    skel_image(skel_image_ids) = 1;
    skel_end_points = bwmorph(skel_image, 'endpoints');
    [y, x] = find(skel_end_points == 1);

    start_x = x(1);
    start_y = y(1);
    end_x = x(length(x));
    end_y = y(length(y));
    
    order = [];

    curr_x = start_x;
    curr_y = start_y;
    last_x = -1;
    last_y = -1;
    
    count = 1;

    ind = sub2ind(image_size,curr_y,curr_x);
    order = [order, ind];

    while count < 1000
        for i = -1:1:1
            for j = -1:1:1
                
                
                % disp(skel_image(curr_x+i,curr_y+j))
                if skel_image(curr_y+j,curr_x+i) == 1 && ~(curr_x+i == last_x && curr_y + j == last_y) && ~(curr_x+i == curr_x && curr_y + j == curr_y)
                    last_x = curr_x;
                    last_y = curr_y;
                    curr_x = curr_x + i;
                    curr_y = curr_y + j;
                    
                    ind = sub2ind(image_size,curr_y,curr_x);
                    order = [order, ind];

                    if curr_x == end_x && curr_y == end_y
                        disp('Correct Target Order Established');
                        return
                    end
                    % disp('fk yeah')
                end
            end
        end
        count = count + 1;
    end

end