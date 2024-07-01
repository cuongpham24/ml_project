# Introduction
To accurately reflect the complexities of the real-world challenges, we will utilize a subset of data from the
2023 Amazon reviews datasets, compiled by McAuley Lab, for our training and testing. As mentioned before, we will randomly select 1 millions sample from each of the three following categories: Home & Kitchen, Books, and Beauty & Personal Care. 

Detailed information about the dataset can be found [here](https://amazon-reviews-2023.github.io/main.html#what-s-new).

# Data Fields
<section id="for-item-metadata">
<h3>For Item Metadata<a class="headerlink" href="#for-item-metadata" title="Permalink to this heading">#</a></h3>
<div class="table-wrapper colwidths-auto docutils container">
<table class="docutils align-default">
<thead>
<tr class="row-odd"><th class="head"><p>Field</p></th>
<th class="head"><p>Type</p></th>
<th class="head"><p>Explanation</p></th>
</tr>
</thead>
<tbody>
<tr class="row-even"><td><p>main_category</p></td>
<td><p>str</p></td>
<td><p>Main category (i.e., domain) of the product.</p></td>
</tr>
<tr class="row-odd"><td><p>title</p></td>
<td><p>str</p></td>
<td><p>Name of the product.</p></td>
</tr>
<tr class="row-even"><td><p>average_rating</p></td>
<td><p>float</p></td>
<td><p>Rating of the product shown on the product page.</p></td>
</tr>
<tr class="row-odd"><td><p>rating_number</p></td>
<td><p>int</p></td>
<td><p>Number of ratings in the product.</p></td>
</tr>
<tr class="row-even"><td><p>features</p></td>
<td><p>list</p></td>
<td><p>Bullet-point format features of the product.</p></td>
</tr>
<tr class="row-odd"><td><p>description</p></td>
<td><p>list</p></td>
<td><p>Description of the product.</p></td>
</tr>
<tr class="row-even"><td><p>price</p></td>
<td><p>float</p></td>
<td><p>Price in US dollars (at time of crawling).</p></td>
</tr>
<tr class="row-odd"><td><p>images</p></td>
<td><p>list</p></td>
<td><p>Images of the product. Each image has different sizes (thumb, large, hi_res). The “variant” field shows the position of image.</p></td>
</tr>
<tr class="row-even"><td><p>videos</p></td>
<td><p>list</p></td>
<td><p>Videos of the product including title and url.</p></td>
</tr>
<tr class="row-odd"><td><p>store</p></td>
<td><p>str</p></td>
<td><p>Store name of the product.</p></td>
</tr>
<tr class="row-even"><td><p>categories</p></td>
<td><p>list</p></td>
<td><p>Hierarchical categories of the product.</p></td>
</tr>
<tr class="row-odd"><td><p>details</p></td>
<td><p>dict</p></td>
<td><p>Product details, including materials, brand, sizes, etc.</p></td>
</tr>
<tr class="row-even"><td><p>parent_asin</p></td>
<td><p>str</p></td>
<td><p>Parent ID of the product.</p></td>
</tr>
<tr class="row-odd"><td><p>bought_together</p></td>
<td><p>list</p></td>
<td><p>Recommended bundles from the websites.</p></td>
</tr>
</tbody>
</table>
</div>
</section>