library(ggplot2)
library(ggdendro)
library(e1071)
library(reshape2)
library(dplyr)

get_confusion <- function(results){
  confusion <- matrix(0, dim(results)[1], dim(results)[2])
  
  recall <- array(0, dim(results)[c(3,1)])
  precision <- array(0, dim(results)[c(3,1)])
  f1 <- array(0, dim(results)[c(3,1)])
  
  accuracy <- 1:dim(results)[3]
  
  for(j in 1:dim(results)[3]){ 
    clasification <- results[,,j]
    support <- Matrix::colSums(clasification)
    
    accuracy[j] <- sum(diag(clasification))/sum(clasification)
    
    recall[j,] <- diag(clasification)/support
    
    precision[j,] <- ifelse(rowSums(clasification)==0, 0, diag(clasification)/rowSums(clasification))
    
    f1[j,] <- 2*ifelse((precision[j,] + recall[j,])==0, 0, precision[j,]*recall[j,]/(precision[j,] + recall[j,]))
    
    jth_confusion <- t(t(clasification)/support)
    confusion <- confusion + jth_confusion
  }
  
  confusion <- confusion/dim(results)[3]
  
  return(list(confusion, accuracy, precision, recall, f1, support))
}

average_scores <-function(precision, support=1, weighted=TRUE){
  score <- array(0, nrow(precision))
  for(j in 1:nrow(precision)){
    if(weighted){
      score[j] <- stats::weighted.mean(precision[j,], support)
    }
    else{
      score[j] <- base::mean(precision[j,])
    }
  }
  return(score)
}

plot_accuracy <- function(accuracies, d=42, TT=42, kernel='kernel', dims=42, 
                          info_type='Info', norm='Norm', gsave=FALSE){
  
  confusionname <- paste('Founders. Test. ', norm, '.', info_type,'. Accuracy=', signif(100*mean(accuracies),3),'%')
  
  p <- ggplot2::ggplot() +
    geom_boxplot(aes(x=100*accuracies), show.legend = FALSE, fill='lightgreen',
                 outlier.size = 0.3, outlier.color = 'maroon') +
    theme(plot.title = element_text(hjust = 0.5, size=11),
          axis.text.y = element_blank(),
          axis.ticks.y = element_blank()) +
    xlab(paste('SVM classification accuracy % (', length(accuracies),' reps)', sep='')) +
    ggtitle(confusionname)
  
  if(gsave == TRUE){
    filename <- paste('accuracy', tolower(gsub(' ', '_', norm)), tolower(info_type), 
                      d, TT, kernel, dims, 'founders', sep='_')
    ggplot2::ggsave(paste(filename, '.png', sep=''), plot=p, device='png', width=6, height=1.2, units='in')
    ggplot2::ggsave(paste(filename, '.pdf', sep=''), plot=p, device='pdf', width=6, height=1.2, units='in')
  }
  
  return(p)
}

plot_confusion <- function(confusion, founders_names, sample_runs=100, hclust_method='single',
                           d=42, TT=42, kernel='kernel', dims=42, info_type='Info', norm='Norm', gsave=FALSE){
  
  rownames(confusion) <- founders_names
  colnames(confusion) <- founders_names
  
  dconf <- confusion/sample_runs
  dconf[lower.tri(dconf)] <- t(dconf)[lower.tri(dconf)]
  diag(dconf) <- 1
  dconf <- 1 - dconf
  hc <- stats::hclust(as.dist(dconf), hclust_method)
  
  test <- as.data.frame(t(confusion)/sample_runs)
  test <- test[hc$order, hc$order]
  colnames(test) <- founders_names[hc$order]
  test$founder <- founders_names[hc$order]
  test$sort <- dim(confusion)[1]:1
  
  test.m <- reshape2::melt(test,id.vars = c("founder", "sort"))
  test.m$founder <- factor(test.m$founder, levels=unique((test.m$founder)[order(test.m$sort)]))
  
  p <- ggplot(test.m,aes(variable, founder)) + geom_tile(aes(fill=value),color = "white") +
    #Creating legend
    guides(fill=guide_colorbar("Accuracy")) +
    #Creating color range
    scale_fill_gradientn(colors=c("royalblue1","yellow","tomato"),
                         guide="colorbar", limits=c(0,1)) +
    #Rotating labels
    theme(axis.text.x = element_text(angle = 270, hjust = 0,vjust=-0.05),
          plot.title = element_text(hjust = 0.5, size=14),
          axis.text = element_text(size=12),
          axis.title = element_text(size=12)) +
    ylab('Labels') + xlab(paste('Predicted (',norm, 
                                ', ', info_type, ')', sep='')) #+ ggtitle(confusionname)
  
  if(gsave==TRUE){
    filename <- paste('confusionSVM', tolower(gsub(' ', '_', norm)), 
                      tolower(info_type), d, TT, kernel, dims, 'founders', sep='_')
    ggplot2::ggsave(paste(filename, '.png', sep=''), plot=p, device='png', width=7.5, height=6.75, units='in')
    ggplot2::ggsave(paste(filename, '.pdf', sep=''), plot=p, device='pdf', width=7.5, height=6.75, units='in')
    
    dhc <- ggdendro::ggdendrogram(hc, rotate=FALSE)
    
    filename <- paste('dendrogram', tolower(gsub(' ', '_', norm)),
                      tolower(info_type), d, TT, kernel, dims, 'founders', sep='_')
    ggplot2::ggsave(paste(filename, '.png', sep=''), plot=dhc, device='png', width=6, height=6, units='in')
    ggplot2::ggsave(paste(filename, '.pdf', sep=''), plot=dhc, device='pdf', width=6, height=6, units='in')
  }
  
  return(p)
}

plot_comparison <- function(conf_diff, founders_names,
                            d=42, TT=42, kernel='kernel', dims=42, info_type='Info', norm='Norm', gsave=FALSE){
  test <- as.data.frame(t(conf_diff))
  colnames(test) <- founders_names
  test$founder <- founders_names
  test$sort <- length(founders_names):1
  
  test.m <- reshape2::melt(test,id.vars = c("founder", "sort"))
  test.m$founder <- factor(test.m$founder, levels=unique((test.m$founder)[order(test.m$sort)]))
  
  
  p <- ggplot(test.m,aes(variable, founder)) + geom_tile(aes(fill=value),color = "white") +
    #Creating legend
    guides(fill=guide_colorbar("Accuracy")) +
    #Creating color range
    scale_fill_gradient2(high = "royalblue1", mid='gray95', low="tomato", midpoint=0,
                         guide="colorbar", limits=c(-0.2, 0.2), oob=scales::squish) +
    #Rotating labels
    theme(axis.text.x = element_text(angle = 270, hjust = 0,vjust=-0.05),
          plot.title = element_text(hjust = 0.5, size=14),
          axis.text = element_text(size=12),
          axis.title = element_text(size=12)) +
    ylab('Labels') + xlab('Predicted') +
    ggtitle(paste('Founders. Test.', info_type, '. Confusion'))
  
  if(gsave==TRUE){
    filename <- paste('comparisonSVM', tolower(gsub(' ', '_', norm)), 
                      gsub(' ', '_', tolower(info_type)), d, TT, kernel, dims, 'founders', sep='_')
    ggplot2::ggsave(paste(filename, '.png', sep=''), plot=p, device='png', width=7.5, height=6.75, units='in')
    ggplot2::ggsave(paste(filename, '.pdf', sep=''), plot=p, device='pdf', width=7.5, height=6.75, units='in')
  }
  
  return(p)
}

compare_descriptor_accuracies <- function(df_original, descriptor='Combined', 
                               d=42, TT=42, kernel='kernel', dims=42, norm='Norm', gsave=FALSE){
  df <- dplyr::arrange(df_original, desc(df_original[descriptor]))
  dfm <- reshape::melt(df, id.var="Line", variable_name='Descriptor')
  dfm$Line <- factor(dfm$Line, levels=dfm$Line[1:nrow(df)])
  titlename <- paste('Classification Average Accuracy (', d, ' directions, ', TT, ' thresholds)', sep='')
  colors <- c('firebrick1', 'blue', 'black')
  p <- ggplot2::ggplot(dfm, aes(x=Line, y=value)) + 
    geom_point(aes(color=Descriptor, shape=Descriptor), size=3) +
    geom_line(aes(color=Descriptor, group=Descriptor), show.legend=TRUE) +
    scale_color_manual(values=colors) +
    theme(plot.title = element_text(hjust = 0.5, vjust = -1, size=15),
          axis.text.x = element_text(size = 9, angle=90),
          axis.text.y = element_text(size = 9),
          axis.title = element_text(size=12)) +
    ylab('Accuracy (%)') +
    ggtitle(titlename) +
    xlab('Barley Line')
  
  if(gsave == TRUE){
    filename <- paste('avg_accuracy', tolower(gsub(' ', '_', norm)), 
                      gsub(' ', '_', tolower(descriptor)), d, TT, kernel, dims, 'founders', sep='_')
    ggplot2::ggsave(paste(filename, '.png', sep=''), plot=p, device='png', width=9, height=3)
    ggplot2::ggsave(paste(filename, '.pdf', sep=''), plot=p, device='pdf', width=9, height=3)
  }
  return(p)
}

unlist_confusion <- function(confusion_list, founders_names, TT){
  confusion <- matrix(0, nrow=length(founders_names), ncol=length(TT))
  rownames(confusion) <- founders_names
  colnames(confusion) <- TT
  
  for(i in 1:length(TT)){
    confusion[,i] <- diag(confusion_list[[i]])
  }
  
  df <- data.frame(confusion)
  colnames(df) <- TT
  df$Line <- founders_names
  
  return(df)
}

compare_threshold_accuracies_total <- function(accuracy_list, TT,
                                               d=42, info_type='info', kernel='kernel', dims=42, norm='Norm', gsave=FALSE){
  df <- data.frame(t(matrix(unlist(accuracy_list), nrow=length(accuracy_list), byrow=TRUE)*100))
  colnames(df) <- as.character(TT)
  
  dfm <- reshape2::melt(df, variable.name='Thresholds', value.name='Accuracy')
  dfm$Thresholds <- factor(dfm$Thresholds, levels=unique(dfm$Thresholds))
  
  confusionname <- paste('Founders. Test. ', norm, '. ', info_type, '.', sep='')
  
  
  p <- ggplot2::ggplot(dfm, aes(x=Thresholds, y=Accuracy, fill=Thresholds)) + 
    ggplot2::geom_boxplot(notch=TRUE) +
    theme(axis.text.x = element_text(angle = 0, hjust = 0,vjust=-0.05),
          plot.title = element_text(hjust = 0.5, size=16),
          axis.text = element_text(size=12),
          axis.title = element_text(size=14)) +
    ylab(paste('SVM classification accuracy % (', length(accuracy_list[[1]]),' reps)', sep='')) +
    xlab(paste('Thresholds (', d, ' directions)', sep='')) +
    ggtitle(confusionname)
  
  if(gsave==TRUE){
    filename <- paste('overall_accuracy', tolower(gsub(' ','_', norm)), 
                      tolower(info_type), d, kernel, dims, 'founders', sep='_')
    
    ggplot2::ggsave(paste(filename, '.pdf', sep=''), plot=p, device='pdf', width=5, height=6)
    ggplot2::ggsave(paste(filename, '.png', sep=''), plot=p, device='png', width=5, height=6)
  }
  
  return(p)
}

compare_direction_accuracies <- function(acc_list1, acc_list2, acc_list3, acc_list4, d1, d2, d3, d4, TT,
                                         info_type='info', kernel='kernel', dims=42, norm='Norm', gsave=FALSE){
  
  df1<- data.frame(t(matrix(unlist(acc_list1), nrow=length(acc_list1), byrow=TRUE)*100))
  df2<- data.frame(t(matrix(unlist(acc_list2), nrow=length(acc_list2), byrow=TRUE)*100))
  df3<- data.frame(t(matrix(unlist(acc_list3), nrow=length(acc_list3), byrow=TRUE)*100))
  df4<- data.frame(t(matrix(unlist(acc_list4), nrow=length(acc_list4), byrow=TRUE)*100))
  colnames(df1) <- as.character(TT)
  colnames(df2) <- as.character(TT)
  colnames(df3) <- as.character(TT)
  colnames(df4) <- as.character(TT)
  
  dfm1 <- reshape2::melt(df1, variable.name='Thresholds', value.name='Accuracy')
  dfm1$Thresholds <- factor(dfm1$Thresholds, levels=unique(dfm1$Thresholds))
  dfm1$Directions <- d1
  dfm2 <- reshape2::melt(df2, variable.name='Thresholds', value.name='Accuracy')
  dfm2$Thresholds <- factor(dfm2$Thresholds, levels=unique(dfm2$Thresholds))
  dfm2$Directions <- d2
  
  dfm3 <- reshape2::melt(df3, variable.name='Thresholds', value.name='Accuracy')
  dfm3$Thresholds <- factor(dfm3$Thresholds, levels=unique(dfm3$Thresholds))
  dfm3$Directions <- d3
  
  dfm4 <- reshape2::melt(df4, variable.name='Thresholds', value.name='Accuracy')
  dfm4$Thresholds <- factor(dfm4$Thresholds, levels=unique(dfm4$Thresholds))
  dfm4$Directions <- d4
  
  dfm <- rbind(dfm1,dfm2,dfm3,dfm4)
  dfm$Directions <- factor(dfm$Directions, levels=sort(c(d1,d2,d3,d4)))
  
  confusionname <- paste('Founders. Test. ', norm, '. ', info_type, '.', sep='')
  
  
  p <- ggplot2::ggplot(dfm, aes(x=Thresholds, y=Accuracy, fill=Directions)) + 
    ggplot2::geom_boxplot(notch=TRUE) +
    theme(axis.text.x = element_text(angle = 0, hjust = 0,vjust=-0.05),
          plot.title = element_text(hjust = 0.5, size=16),
          axis.text = element_text(size=12),
          axis.title = element_text(size=14)) +
    ylab(paste('SVM classification accuracy % (', length(acc_list1[[1]]),' reps)', sep='')) +
    xlab(paste('Thresholds')) +
    ggtitle(confusionname)
  
  if(gsave == TRUE){
    filename <- paste('overall_accuracy_dirs', tolower(gsub(' ','_', norm)), 
                      tolower(info_type), kernel, dims, 'founders', sep='_')
    
    ggplot2::ggsave(paste(filename, '.pdf', sep=''), plot=p, device='pdf', width=10, height=6)
    ggplot2::ggsave(paste(filename, '.png', sep=''), plot=p, device='png', width=10, height=6)
  }
  return(p)
}

compare_threshold_accuracies_lines <- function(df_original, descriptor='16',
                                         d=42, info_type='info', kernel='kernel', dims=42, norm='Norm', gsave=FALSE){
  
  df <- dplyr::arrange(df_original, desc(df_original[descriptor]))
  dfm <- reshape2::melt(df, id.vars='Line', variable.name='Thresholds', value.name='value')
  dfm$Thresholds <- factor(dfm$Thresholds, levels=TT)
  dfm$Line <- factor(dfm$Line, levels=unique(dfm$Line))
  
  titlename <- paste('Classification Average Accuracy (', info_type, ' descriptors, ', d, ' directions)', sep='')
  
  p <- ggplot2::ggplot(dfm, aes(x=Line, y=value)) + 
    geom_point(aes(color=Thresholds, shape=Thresholds), size=3) +
    geom_line(aes(color=Thresholds, group=Thresholds), show.legend=TRUE) +
    #    scale_color_manual(values=colors) +
    theme(plot.title = element_text(hjust = 0.5, vjust = -1, size=15),
          axis.text.x = element_text(size = 9, angle=90),
          axis.text.y = element_text(size = 9),
          axis.title = element_text(size=12)) +
    ylab('Accuracy (%)') +
    ggtitle(titlename) +
    xlab('Barley Line')
  
  if(gsave == TRUE){
    filename <- paste('overall_avg_accuracy', tolower(gsub(' ', '_', norm)), 
                      tolower(info_type), d, kernel, dims, 'founders', sep='_')
    ggplot2::ggsave(paste(filename,'.png', sep=''), plot=p, device='png', width=9, height=3)
    ggplot2::ggsave(paste(filename,'.pdf', sep=''), plot=p, device='pdf', width=9, height=3)
  }
  return(p)
}