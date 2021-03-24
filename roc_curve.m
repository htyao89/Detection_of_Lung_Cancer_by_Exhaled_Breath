function auc = roc_curve(deci,label_y,train)
    label_y=(label_y+0.5)*2;
    deci = (deci-min(deci))/(max(deci)-min(deci));
    [val,ind]=sort(deci,'descend');
    roc_y = label_y(ind);
    stack_x = cumsum(roc_y==-1)/sum(roc_y==-1);
    stack_y = cumsum(roc_y==1)/sum(roc_y==1);
    auc = sum((stack_x(2:length(roc_y),1)-stack_x(1:length(roc_y)-1,1)).*stack_y(2:length(roc_y),1));
    plot(stack_x,stack_y);
    xlabel('False Positive Rate');
    ylabel('True Positive Rate');
    if train==1
        title([' ROC curve of (AUC=' num2str(auc) ')']);
    else
        title(['ROC curve of (AUC=' num2str(auc) ')']);
    end
end