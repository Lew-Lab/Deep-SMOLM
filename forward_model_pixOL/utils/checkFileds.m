function checkFileds(s, fileds)

for i = 1:numel(fileds)

    if ~isfield(s, fileds{i})
        error(strcat('field ', fileds{i}, ' is not defined.'))
    end
end