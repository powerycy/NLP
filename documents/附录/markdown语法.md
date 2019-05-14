---
description: Markdown语法
---

# Markdown语法

# Heading 1

## Heading 2

### Heading 3

#### Heading 4

##### Heading 5

###### Heading 6

Paragraph

**加粗**

_斜体_

~~穿透线~~

[链接](http://www.baidu.com)

![百度logo](https://www.baidu.com/img/superlogo_c4d7df0a003d3db9b65e9ef0fe6da1ec.png?where=super)

* a
* b
* c



1. a
2. b
3. c

* [ ] a
* [ ] b
* [ ] c

> 引用的东东
>
> 这是表示在引用别人的东西



```
var a = "this is a code mark";
console.log(a);
```

    var a = "this is a code mark";
    console.log(a);


$$x = y;x^2+y^2=1$$

你好吗？ [^1] 嗯，很好的呢 [^2]



| 姓名   | 年龄   | 性别   |
| :--- | :--- | :--- |
| 老罗   | 28   | 男    |
| 胡杰   | 28   | 男    |



接下来是一个分割线

---



```
​```mermaid
%% Example with slection of syntaxes
        gantt
        dateFormat  YYYY-MM-DD
        title Adding GANTT diagram functionality to mermaid

        section A section
        Completed task            :done,    des1, 2014-01-06,2014-01-08
        Active task               :active,  des2, 2014-01-09, 3d
        Future task               :         des3, after des2, 5d
        Future task2               :         des4, after des3, 5d

        section Critical tasks
        Completed task in the critical line :crit, done, 2014-01-06,24h
        Implement parser and jison          :crit, done, after des1, 2d
        Create tests for parser             :crit, active, 3d
        Future task in critical line        :crit, 5d
        Create tests for renderer           :2d
        Add to mermaid                      :1d

        section Documentation
        Describe gantt syntax               :active, a1, after des1, 3d
        Add gantt diagram to demo page      :after a1  , 20h
        Add another diagram to demo page    :doc1, after a1  , 48h

        section Last section
        Describe gantt syntax               :after doc1, 3d
        Add gantt diagram to demo page      : 20h
        Add another diagram to demo page    : 48h
​```
```



[^1]: 这是注解1

[^2]: 这是注解2
